#include "Mcmc.hpp"
#include "analysis/Particle.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <functional>

Mcmc::Mcmc(void) {}

double Mcmc::run(Particle& model, int iterations, bool tune, bool debug, int tuneFrequency, int printFrequency, double tempering, bool updateInvar){
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution unif(0.0,1.0);

    model.refreshLikelihood();
    double currentLikelihood = model.lnLikelihood() * tempering;
    double currentPrior = model.lnPrior();
    double currentPosterior = currentLikelihood + currentPrior;

    bool sampleAlpha = model.getNumRates() > 1;

    for(int i = 0; i < iterations; i++){
        double moveChoice = unif(generator);

        std::function<double(Particle&)> proposal;
        int numGibbs = 2;

        if(moveChoice < 0.2){
            proposal = [](Particle& m){ return m.topologyMove(); };
            numGibbs = model.getNumNodes()/4;
        }
        else if(moveChoice < 0.3 && sampleAlpha){
            proposal = [](Particle& m){ return m.shapeMove();} ;
        }
        else if(moveChoice < 0.6){
            proposal = [this](Particle& m){ return m.branchMove(); };
            numGibbs = model.getNumNodes()/4;
        }
        else if(moveChoice < 0.7 && updateInvar){
            proposal = [this](Particle& m){ return m.invarMove(); };
        }
        else{
            proposal = [this](Particle& m){ return m.stationaryMove(); };
            numGibbs = model.getNumCategories();
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
                std::cout << i << "\t" << currentPosterior << "\t" << currentLikelihood << "\t" << currentPrior << 
                    "\tBranch Scale Rate:" << (double)model.acceptedBranchLength/(double)model.proposedBranchLength << 
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