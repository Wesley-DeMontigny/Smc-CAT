#include <iostream>
#include <iomanip>
#include <random>
#include <format>
#include <omp.h>
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
    int numParticles = 1000;
    int numThreads = omp_get_max_threads();
    bool invar = false;

    Alignment aln("/workspaces/FastCAT/local_testing/globin_test.fasta");
    int numChar = aln.getNumChar();

    std::cout << std::format("Utilizing {} threads for working with {}", numThreads, numParticles) << std::endl;
    std::vector<Particle> threadParticles;
    threadParticles.reserve(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        threadParticles.emplace_back(aln);
    }

    Mcmc analysis{};

    std::vector<int> oldShardAssignments(numParticles, 0);
    std::vector<int> currentShardAssignments(numParticles, 0);
    std::vector<double> rawWeights(numParticles, -1.0 * std::log(numParticles));

    std::mt19937 gen = std::mt19937(std::random_device{}());
    std::uniform_real_distribution systematicUnif(0.0, 1.0/numParticles);

        std::vector<double> rawLogLikelihoods(numParticles, 0);

    // Particle initialization
    std::cout << "Initializing SMC..." << std::endl;
    Particle& initP = threadParticles[0];
    for(int n = 0; n < numParticles; ++n){
        if(!invar)
            initP.setInvariance(0.0);
        
        initP.write(n, "/workspaces/FastCAT/local_testing/particle_states.0.s0.h5");
        rawLogLikelihoods[n] = initP.lnLikelihood();
        initP.initialize();
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

            // Systematic resamplign scheme (has less variance than multinomial resampling)
            double u = systematicUnif(gen);
            std::vector<int> assignments;
            assignments.reserve(numParticles);
            double cumulative = normalizedWeights[0];
            int currentWeight = 0;
            double ruler = u;
            double increment = 1.0/numParticles;
            while(assignments.size() < numParticles){
                if(ruler <= cumulative){
                    assignments.push_back(currentWeight);
                    ruler += increment;
                }
                else {
                    cumulative += normalizedWeights[++currentWeight];
                }
            }

            std::cout << "Rejuvinating Particles..." << std::endl;
            #pragma omp parallel for schedule(dynamic)
            for (int n = 0; n < numParticles; ++n) {
                int threadID = omp_get_thread_num();
                Particle& p = threadParticles[threadID];
                int particleID = assignments[n];

                std::string lastFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.s{}.h5", lastID, oldShardAssignments[particleID]);
                std::string newFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.s{}.h5", i, threadID);

                p.read(particleID, lastFile);

                analysis.run(p, 10, false, false, 1, 1, currentTemp, invar);
                p.gibbsPartitionMove(currentTemp);
                p.refreshLikelihood();
                p.accept();
                rawLogLikelihoods[n] = p.lnLikelihood();
                rawWeights[n] = -1.0 * std::log(numParticles);

                currentShardAssignments[n] = threadID;
                p.write(n, newFile);
            }
            lastID = i;
            std::swap(currentShardAssignments, oldShardAssignments);
            
            for(Particle& p : threadParticles){
                p.tune();
            }
        }

        lastTemp = currentTemp;
    }
    
    int maxID = 0;
    double maxW = -1.0 * INFINITY;
    for(int i = 0; i < numParticles; i++){
        if(rawWeights[i] > maxW){
            maxID = i;
            maxW = rawWeights[i];
        }
    }

    // Generate posterior statistics
    
    return 0;
}