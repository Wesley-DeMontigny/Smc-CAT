#include <iostream>
#include <iomanip>
#include "core/Settings.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/Particle.hpp"
#include "core/Alignment.hpp"
#include "analysis/Mcmc.hpp"
#include <random>
#include <format>

void computeNextStep(std::vector<double>& rawWeights, std::vector<double>& rawLogLikelihoods, 
                     std::vector<double>& normalizedWeights, double& ESS, const double& lastESS,
                     double& currentTemp, const double& lastTemp, const int& numParticles){
    double maxW = *std::max_element(rawWeights.begin(), rawWeights.end());

    double upperThresh = 0.8 * lastESS;

    do {
        currentTemp += 0.005;
        if(currentTemp > 1.0)
            currentTemp = 1.0;

        double total = 0.0;
        for(int n = 0; n < numParticles; n++){
            rawWeights[n] = rawWeights[n] + (currentTemp - lastTemp) * rawLogLikelihoods[n]; // Adjust the weights
            double adjustedWeight = std::exp(rawWeights[n] - maxW); // Start normalization procedure to compute ESS
            normalizedWeights[n] = adjustedWeight;
            total += adjustedWeight;
        }

        double weightTotal = 0.0;
        for(int n = 0; n < numParticles; ++n){
            normalizedWeights[n] /= total;
            weightTotal += normalizedWeights[n] * normalizedWeights[n];
        }

        ESS = 1.0/weightTotal;
    }
    while(ESS > upperThresh && currentTemp != 1.0);
}

int main() {
    Alignment aln("/workspaces/FastCAT/local_testing/globin_test.fasta");

    Particle p(aln);
    Mcmc analysis{};

    int numParticles = 100;
    std::vector<double> rawWeights(numParticles, -1.0 * std::log(numParticles));
    std::vector<double> rawLogLikelihoods(numParticles, 0);

    std::mt19937 gen = std::mt19937(std::random_device{}());
    std::uniform_real_distribution unif(0.0, 1.0);

    // Particle initialization
    std::cout << "Initializing SMC..." << std::endl;
    for(int n = 0; n < numParticles; ++n){
        p.write(n, "/workspaces/FastCAT/local_testing/particle_states.0.h5");
        rawLogLikelihoods[n] = p.lnLikelihood();
        p.initialize();
    }
    std::cout << "Initialized Particles" << std::endl;

    // SMC algorithm
    std::cout << "Starting SMC..." << std::endl;
    int lastID = 0;
    double lastTemp = 0.0;
    double lastESS = numParticles;
    for(int i = 1; lastTemp < 1.0; ++i){
        std::string lastFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.h5", lastID);
        std::string newFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.h5", i);

        std::vector<double> normalizedWeights(numParticles, 0.0);
        double currentTemp = lastTemp;
        double ESS = 0.0;
        computeNextStep(rawWeights, rawLogLikelihoods, normalizedWeights, ESS, lastESS, currentTemp, lastTemp, numParticles);

        std::cout << std::format("{}\tTemp: {:.3f}\t ESS: {:.3f}", i, currentTemp, ESS) << std::endl;

        // This part is trivially parallelizable and we should look into it.
        // I don't think we want to run MCMC on the last re-weighing
        if(ESS <= 0.5 * numParticles && currentTemp != 1.0){
            std::cout << "Starting Resampling and Rejuvination" << std::endl;
            for(int n = 0; n < numParticles; ++n){
                double draw = unif(gen);
                int particle_id = numParticles-1;
                double cumulative = 0.0;
                for(int j = 0; j < numParticles; ++j){
                    cumulative += normalizedWeights[j];
                    if(draw < cumulative){
                        particle_id = j;
                        break;
                    }
                }

                p.read(particle_id, lastFile);
                analysis.run(p, 10, false, false, 1, 1, currentTemp);
                rawLogLikelihoods[n] = p.lnLikelihood();
                std::cout << std::format("Resampled and Rejuvinated Particle {}", n) << std::endl;
                p.write(n, newFile);
                rawWeights[n] = -1.0 * std::log(numParticles);
            }
            lastID = i;
            lastESS = numParticles;
        }
        else{
            lastESS = ESS;
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

    std::string lastFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.h5", lastID);
    p.read(maxID, lastFile);
    std::cout << p.getNumCategories() << std::endl;
    std::cout << p.getNewick() << std::endl;

    
    return 0;
}