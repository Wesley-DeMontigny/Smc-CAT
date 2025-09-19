#include "Mcmc.hpp"
#include "analysis/Particle.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <functional>

Mcmc::Mcmc(void) {}

double Mcmc::run(Particle& model, int iterations, bool tune, bool debug, int tuneFrequency, int printFrequency, double tempering){
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution unif(0.0,1.0);

    double currentLikelihood = model.lnLikelihood() * tempering;
    double currentPrior = model.lnPrior();
    double currentPosterior = currentLikelihood + currentPrior;

    for(int i = 0; i < iterations; i++){
        double moveChoice = unif(generator);

        std::function<double(Particle&)> proposal;
        int numGibbs = 5;

        if(moveChoice < 0.25){
            proposal = [](Particle& m){ return m.topologyMove();};
        }
        else if(moveChoice < 0.5){
            proposal = [this](Particle& m){ return m.branchMove();};
        }
        else if(moveChoice < 0.55){
            proposal = [this](Particle& m){ return m.gibbsPartitionMove();};
            numGibbs = 1;
        }
        else{
            proposal = [this](Particle& m){ return m.stationaryMove();};
        }

        for(int j = 0; j < numGibbs; j++){
            double hastings = proposal(model);
            model.refreshLikelihood();

            double newPrior = model.lnPrior();
            double newLikelihood = model.lnLikelihood() * tempering;
            double newPosterior = newPrior + newLikelihood;
            double posteriorRatio = newPosterior - currentPosterior + hastings;

            if(std::log(unif(generator)) <= posteriorRatio){
                model.accept();
                currentPosterior = newPosterior;
                currentPrior = newPrior;
                currentLikelihood = newLikelihood;
            }
            else{
                model.reject();
            }
        }

        if(debug){
            if(i % printFrequency == 0){
                std::cout << i << "\t" << currentPosterior << "\tBranch Scale Rate:" << (double)model.acceptedBranchLength/(double)model.proposedBranchLength << 
                    "\tNNI Rate:" << (double)model.acceptedNNI/(double)model.proposedNNI << "\tStationary Rate:" << (double)model.acceptedStationary/(double)model.proposedStationary <<
                    "\tSubtree Scale Rate:" << (double)model.acceptedSubtreeScale/(double)model.proposedSubtreeScale << std::endl;
            }
        }
        if(tune){
            if(i % tuneFrequency == 0){
                model.tune();
            }
        }
    }

    return currentPosterior;
}