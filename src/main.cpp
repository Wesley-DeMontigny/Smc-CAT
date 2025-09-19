#include <iostream>
#include "core/Settings.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/Particle.hpp"
#include "core/Alignment.hpp"
#include "analysis/Mcmc.hpp"
#include <random>
#include <format>

int main() {
    Alignment aln("/workspaces/FastCAT/local_testing/globin_test.fasta");

    Particle p(aln);
    Mcmc analysis{};

    int numParticles = 50;
    std::vector<double> weights;
    std::vector<double> logPost;

    std::mt19937 gen = std::mt19937(std::random_device{}());
    std::uniform_real_distribution unif(0.0, 1.0);

    // Particle initialization
    std::cout << "Initializing SMC..." << std::endl;
    for(int n = 0; n < numParticles; ++n){
        double final = analysis.run(p, 10, false, false, 50, 100, 0.0); // Sample entirely from the prior because of the temperature
        logPost.push_back(final);
        std::cout << "Initialized Particle " << n << std::endl;
        p.write(n, "/workspaces/FastCAT/local_testing/particle_states.0.h5");
        p.initialize();
    }

    double maxP = *std::max_element(logPost.begin(), logPost.end());
    double total = 0.0;
    for(double& d : logPost){
        d -= maxP;
        d = std::exp(d);
        total += d;
    }
    for(int n = 0; n < numParticles; ++n){
        double newWeight = logPost[n]/total;
        weights.push_back(newWeight);
    }

    // SMC algorithm
    std::cout << "Starting SMC..." << std::endl;
    for(int i = 1; i <= 100; ++i){
        std::string lastFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.h5", i-1);
        std::string newFile = std::format("/workspaces/FastCAT/local_testing/particle_states.{}.h5", i);

        double currentTemp = std::pow(i/100.0, 2);
        double lastTemp = std::pow((i-1)/100.0, 2);

        logPost.clear();
        for(int n = 0; n < numParticles; ++n){
            double draw = unif(gen);
            int particle_id = numParticles-1;
            double cumulative = 0.0;
            for(int j = 0; j < numParticles; ++j){
                cumulative += weights[j];
                if(draw < cumulative){
                    particle_id = j;
                    break;
                }
            }

            p.read(particle_id, lastFile);
            double final = analysis.run(p, 100, true, false, 50, 100, currentTemp); // Geometric tempering
            logPost.push_back(final);
            std::cout << std::format("Resampled and Rejuvinated Particle {} ({})", n, final) << std::endl;
            p.write(n, newFile);
        }

        double maxP = *std::max_element(logPost.begin(), logPost.end());
        double total = 0.0;
        for(double& d : logPost){
            d -= maxP;
            d = std::exp(d);
            total += d;
        }

        weights.clear();
        for(int n = 0; n < numParticles; ++n){
            double newWeight = logPost[n]/total;
            weights.push_back(newWeight);
        }
    }
    
    return 0;
}