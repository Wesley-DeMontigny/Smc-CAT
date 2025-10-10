#include <iostream>
#include <iomanip>
#include <random>
#include <format>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include "core/Settings.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/Particle.hpp"
#include "core/Alignment.hpp"
#include "analysis/Mcmc.hpp"

void computeNextStep(std::vector<double>& rawWeights, std::vector<double>& rawLogLikelihoods, 
                     std::vector<double>& normalizedWeights, double& ESS,
                     double& currentTemp, const double& lastTemp, const int& numParticles) {
    double targetESS = 0.6 * numParticles;
    double low = lastTemp;
    double high = 1.0;

    auto computeESS = [&](double temp) {
        std::vector<double> tempWeights(numParticles);
        double total = 0.0;

        for (int n = 0; n < numParticles; n++) {
            tempWeights[n] = rawWeights[n] + (temp - lastTemp) * rawLogLikelihoods[n];
        }

        double maxW = *std::max_element(tempWeights.begin(), tempWeights.end());

        std::vector<double> expWeights(numParticles);
        for (int n = 0; n < numParticles; n++) {
            expWeights[n] = std::exp(tempWeights[n] - maxW);
            total += expWeights[n];
        }

        double sumSq = 0.0;
        for (int n = 0; n < numParticles; n++) {
            expWeights[n] /= total;
            sumSq += expWeights[n] * expWeights[n];
        }

        return 1.0 / sumSq;
    };

    // Binary search for temperature
    while (high - low > 1e-5) {
        double mid = 0.5 * (low + high);
        double midESS = computeESS(mid);

        if (midESS > targetESS) {
            low = mid;   // We can still increase temperature
        } else {
            high = mid;  // Too much degeneracy
        }
    }

    currentTemp = high;

    for (int n = 0; n < numParticles; n++) {
        rawWeights[n] = rawWeights[n] + (currentTemp - lastTemp) * rawLogLikelihoods[n];
    }

    double maxW = *std::max_element(rawWeights.begin(), rawWeights.end());
    double total = 0.0;
    for (int n = 0; n < numParticles; n++) {
        normalizedWeights[n] = std::exp(rawWeights[n] - maxW);
        total += normalizedWeights[n];
    }

    double sumSq = 0.0;
    for (int n = 0; n < numParticles; n++) {
        normalizedWeights[n] /= total;
        sumSq += normalizedWeights[n] * normalizedWeights[n];
    }

    ESS = 1.0 / sumSq;
}

int main() {
    int numParticles = 500;
    int rejuvinationIterations = 20;
    int numThreads = omp_get_max_threads();
    bool invar = true;
    bool serialize = false;
    int numRates = 3;

    std::mt19937 gen = std::mt19937(std::random_device{}());
    std::uniform_real_distribution systematicUnif(0.0, 1.0/numParticles);
    std::vector<double> rawLogLikelihoods(numParticles, 0);
    std::vector<double> rawWeights(numParticles, -1.0 * std::log(numParticles));
    std::vector<int> oldShardAssignments(numParticles, 0);
    std::vector<int> currentShardAssignments(numParticles, 0);

    Alignment aln("/workspaces/FastCAT/local_testing/globin_test.fasta");
    int numChar = aln.getNumChar();

    std::cout << std::format("Utilizing {} threads for working with {}", numThreads, numParticles) << std::endl;

    Mcmc mcmc{};
    std::vector<Particle> particles;

    std::chrono::steady_clock::time_point preAnalysis = std::chrono::steady_clock::now();


    if(serialize){
        particles.reserve(numThreads);
        for (int t = 0; t < numThreads; ++t) {
            particles.emplace_back(Particle(aln, numRates, invar));
        }

        // Particle initialization
        std::cout << "Initializing SMC..." << std::endl;
        Particle& initP = particles[0];
        for(int n = 0; n < numParticles; ++n){  
            initP.write(n, "/workspaces/FastCAT/local_testing/particle_states.0.s0.h5");
            rawLogLikelihoods[n] = initP.lnLikelihood();
            initP.initialize(invar);
        }
        std::cout << "Initialized Particles..." << std::endl;

        // SMC algorithm
        std::cout << "Starting SMC..." << std::endl;
        int lastID = 0;
        double lastTemp = 0.0;
        for(int i = 1; lastTemp < 1.0; ++i){
            std::vector<double> normalizedWeights(numParticles, 0.0);
            double currentTemp = lastTemp;
            double ESS = 0.0;
            computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, currentTemp, lastTemp, numParticles);

            std::cout << std::format("{}\tTemp: {:.5f}\t ESS: {:.5f}", i, currentTemp, ESS) << std::endl;

            if(ESS <= 0.6 * numParticles && currentTemp != 1.0){
                std::cout << "Resampling Particles..." << std::endl;

                // Systematic resampling
                double u = systematicUnif(gen);
                std::vector<int> assignments;
                assignments.reserve(numParticles);
                double cumulative = normalizedWeights[0];
                int currentWeight = 0;
                int k = 0;
                double ruler = u;
                double step = 1.0 / numParticles;
                while (static_cast<int>(assignments.size()) < numParticles) {
                    if (ruler <= cumulative) {
                        assignments.push_back(k);
                        ruler += step;
                    } else {
                        if (++k >= numParticles) { k = numParticles - 1; cumulative = 1.0; }
                        else cumulative += normalizedWeights[k];
                    }
                }

                std::cout << "Rejuvinating Particles..." << std::endl;
                #pragma omp parallel for schedule(dynamic)
                for (int n = 0; n < numParticles; ++n) {
                    int threadID = omp_get_thread_num();
                    Particle& p = particles[threadID];
                    int particleID = assignments[n];

                    std::string lastFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.s{}.h5", lastID, oldShardAssignments[particleID]);
                    std::string newFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.s{}.h5", i, threadID);

                    p.read(particleID, lastFile);

                    mcmc.run(p, rejuvinationIterations, false, false, 1, 1, currentTemp, invar);
                    //p.gibbsPartitionMove(currentTemp);
                    //p.refreshLikelihood();
                    //p.accept();
                    rawLogLikelihoods[n] = p.lnLikelihood();
                    rawWeights[n] = -1.0 * std::log(numParticles);

                    currentShardAssignments[n] = threadID;
                    p.write(n, newFile);
                }
                lastID = i;
                std::swap(currentShardAssignments, oldShardAssignments); 
                for(Particle& p : particles)
                    p.tune();
            }

            lastTemp = currentTemp;
        }
    }
    else {
        particles.reserve(numParticles);
        for (int t = 0; t < numParticles; ++t) {
            particles.emplace_back(aln, numRates, invar);
        }

        // Particle initialization
        std::cout << "Initializing SMC..." << std::endl;
        for(int n = 0; n < numParticles; ++n){
            rawLogLikelihoods[n] = particles[n].lnLikelihood();
        }
        std::cout << "Initialized Particles..." << std::endl;

        // SMC algorithm
        std::cout << "Starting SMC..." << std::endl;
        double lastTemp = 0.0;
        for(int i = 1; lastTemp < 1.0; ++i){
            std::vector<double> normalizedWeights(numParticles, 0.0);
            double currentTemp = lastTemp;
            double ESS = 0.0;
            computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, currentTemp, lastTemp, numParticles);

            std::cout << std::format("{}\tTemp: {:.5f}\t ESS: {:.5f}", i, currentTemp, ESS) << std::endl;

            if(ESS <= 0.6 * numParticles && currentTemp != 1.0){
                std::cout << "Resampling Particles..." << std::endl;

                // Systematic resampling
                std::vector<int> assignments;
                assignments.reserve(numParticles);

                double u = systematicUnif(gen);
                double cumulative = normalizedWeights[0];
                int k = 0;
                double ruler = u;
                double step = 1.0 / numParticles;
                while (static_cast<int>(assignments.size()) < numParticles) {
                    if (ruler <= cumulative) {
                        assignments.push_back(k);
                        ruler += step;
                    } else {
                        if (++k >= numParticles) { k = numParticles - 1; cumulative = 1.0; }
                        else cumulative += normalizedWeights[k];
                    }
                }

                std::vector<int> counts(numParticles, 0);
                for (int idx : assignments) ++counts[idx];

                // Identify "dead" particles
                std::vector<int> freeSlots;
                for (int i = 0; i < numParticles; ++i) {
                    if (counts[i] == 0) freeSlots.push_back(i);
                }

                // Reuse dead particles
                for (int i = 0; i < numParticles; ++i) {
                    if (counts[i] > 1) {
                        counts[i]--;
                        while (counts[i] > 0) {
                            int target = freeSlots.back();
                            freeSlots.pop_back();
                            particles[target].copy(particles[i]);
                            counts[i]--;
                        }
                    }
                }

                std::cout << "Rejuvinating Particles..." << std::endl;
                #pragma omp parallel for schedule(dynamic)
                for (int n = 0; n < numParticles; ++n) {
                    Particle& p = particles[n];

                    mcmc.run(p, rejuvinationIterations, false, false, 1, 1, currentTemp, invar);
                    //p.gibbsPartitionMove(currentTemp);
                    //p.refreshLikelihood();
                    //p.accept();
                    rawLogLikelihoods[n] = p.lnLikelihood();
                    rawWeights[n] = -1.0 * std::log(numParticles);
                }
            }

            lastTemp = currentTemp;
        }
    }

    std::cout << "Computing Maximum Clade Consensus Tree..." << std::endl;

    std::vector<double> normalizedWeights(numParticles, 0.0);
    double maxW = *std::max_element(rawWeights.begin(), rawWeights.end());
    double total = 0.0;
    for (int n = 0; n < numParticles; n++) {
        normalizedWeights[n] = std::exp(rawWeights[n] - maxW);
        total += normalizedWeights[n];
    }

    for (int n = 0; n < numParticles; n++) {
        normalizedWeights[n] /= total;
    }

    std::unordered_map<std::string, double> splitPosteriorProbabilities;
    std::vector<std::set<std::string>> particleSplits;
    if(!serialize){
        // Get split posterior probabilities
        for(int i = 0; i < particles.size(); i++){
            std::set<std::string> splitStrings = particles[i].getSplits();
            particleSplits.push_back(splitStrings);
            for(std::string split : splitStrings){
                if (splitPosteriorProbabilities.count(split)) {
                    splitPosteriorProbabilities[split] += normalizedWeights[i];
                }
                else {
                    splitPosteriorProbabilities[split] = normalizedWeights[i];
                }
            }
        }
    }
    else{

    }

    std::vector<std::set<std::string>> deduped;
    std::set<std::set<std::string>> seen;
    for (auto& s : particleSplits)
        seen.insert(s);
    deduped.assign(seen.begin(), seen.end());

    std::cout << std::format("There were {} unique topologies in the final approximation", deduped.size()) << std::endl;

    // Get max score
    int maxID = 0;
    double maxScore = -INFINITY;
    for(int i = 0; i < particles.size(); i++){
        double currentScore = 0.0;
        for(std::string s : particleSplits[i]){
            currentScore += std::log(splitPosteriorProbabilities[s]);
        }

        if(currentScore > maxScore){
            maxScore = currentScore;
            maxID = i;
        }
    }

    if(!serialize){
        std::cout << particles[maxID].getNewick() << std::endl;
        std::vector<double> rates = particles[maxID].getRates();
        for(auto r : rates)
            std::cout << r << std::endl;
    }
    else {

    }

    std::chrono::steady_clock::time_point postAnalysis = std::chrono::steady_clock::now();
    std::cout << "The analysis completed in " << std::chrono::duration_cast<std::chrono::minutes>(postAnalysis - preAnalysis).count() << "[minutes]" << std::endl;
    
    return 0;
}