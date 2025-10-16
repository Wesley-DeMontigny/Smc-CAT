#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include "core/Settings.hpp"
#include "analysis/SerializedParticle.hpp"
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
    int rejuvinationIterations = 5;
    int numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);
    bool invar = false;
    int numRates = 4;

    std::mt19937 gen = std::mt19937(std::random_device{}());
    std::uniform_real_distribution unif(0.0, 1.0);
    std::uniform_real_distribution systematicUnif(0.0, 1.0/numParticles);
    
    std::vector<double> rawLogLikelihoods(numParticles, 0);
    std::vector<double> rawWeights(numParticles, -1.0 * std::log(numParticles));

    std::unordered_map<std::string, double> splitPosteriorProbabilities;
    std::vector<std::set<std::string>> particleSplits;
    particleSplits.reserve(numParticles);
    for(int i = 0; i < numParticles; i++){
        particleSplits.push_back({});
    }

    std::vector<SerializedParticle> currentSerializedParticles;
    std::vector<SerializedParticle> oldSerializedParticles;
    currentSerializedParticles.reserve(numParticles);
    oldSerializedParticles.reserve(numParticles);

    Alignment aln("/workspaces/FastCAT/local_testing/globin_aa_aligned.fasta");
    int numChar = aln.getNumChar();

    std::cout << "Utilizing " << numThreads << " threads for working with " << numParticles << std::endl;

    Mcmc mcmc{};
    std::vector<Particle> particles;

    std::chrono::steady_clock::time_point preAnalysis = std::chrono::steady_clock::now();

    particles.reserve(numThreads);
    for (int t = 0; t < numThreads; t++) {
        particles.emplace_back(aln, numRates, invar);
    }

    // Particle initialization
    std::cout << "Initializing SMC..." << std::endl;
    Particle& initP = particles[0];
    for(int n = 0; n < numParticles; n++){  
        rawLogLikelihoods[n] = initP.lnLikelihood();
        currentSerializedParticles.emplace_back(
            initP.getNewick(), initP.getAssignments(), initP.getCategories(),
            initP.getBaseMatrix(), initP.getPInvar(), initP.getShape()
        );
        particleSplits[n] = initP.getSplits();
        initP.initialize(invar);
    }
    oldSerializedParticles = currentSerializedParticles;
    std::cout << "Initialized Particles..." << std::endl;

    // SMC algorithm
    std::cout << "Starting SMC..." << std::endl;
    double lastTemp = 0.0;
    for(int i = 1; lastTemp < 1.0; i++){
        std::vector<double> normalizedWeights(numParticles, 0.0);
        double currentTemp = lastTemp;
        double ESS = 0.0;
        computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, currentTemp, lastTemp, numParticles);

        // Generate split posterior probabilities
        splitPosteriorProbabilities.clear();
        for(int n = 0; n < numParticles; n++){
            for(std::string split : particleSplits[n]){
                if (splitPosteriorProbabilities.count(split)) {
                    splitPosteriorProbabilities[split] += normalizedWeights[i];
                }
                else {
                    splitPosteriorProbabilities[split] = normalizedWeights[i];
                }
            }
        }

        std::cout << i << "\tTemp: " << currentTemp << "\tESS: " << ESS << std::endl;

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
                } 
                else {
                    if (++k >= numParticles) { k = numParticles - 1; cumulative = 1.0; }
                    else cumulative += normalizedWeights[k];
                }
            }

            std::cout << "Rejuvenating Particles..." << std::endl;
            #pragma omp parallel for schedule(dynamic)
            for (int n = 0; n < numParticles; n++) {
                int threadID = omp_get_thread_num();
                Particle& p = particles[threadID];
                int particleID = assignments[n];
                p.copyFromSerialized(currentSerializedParticles[particleID]);

                mcmc.run(p, rejuvinationIterations, splitPosteriorProbabilities, invar, currentTemp);

                if(unif(gen) < 0.05){
                    //p.gibbsPartitionMove(currentTemp);
                    //p.refreshLikelihood();
                    //p.accept();
                }
                rawLogLikelihoods[n] = p.lnLikelihood();
                rawWeights[n] = -1.0 * std::log(numParticles);
                p.writeToSerialized(oldSerializedParticles[n]);

                particleSplits[n] = p.getSplits();
            }
            std::swap(currentSerializedParticles, oldSerializedParticles);
        }

        lastTemp = currentTemp;
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

        if(currentScore > maxScore || (currentScore == maxScore && normalizedWeights[i] > normalizedWeights[maxID])){
            maxScore = currentScore;
            maxID = i;
        }
    }

    std::cout << currentSerializedParticles[maxID].newick << std::endl;

    std::chrono::steady_clock::time_point postAnalysis = std::chrono::steady_clock::now();
    std::cout << "The analysis completed in " << std::chrono::duration_cast<std::chrono::minutes>(postAnalysis - preAnalysis).count() << "[minutes]" << std::endl;
    
    return 0;
}