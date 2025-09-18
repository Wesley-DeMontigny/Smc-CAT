#include "Mcmc.hpp"
#include "core/Alignment.hpp"
#include "analysis/Model.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <functional>

Mcmc::Mcmc(Alignment& aln, Model& m) : alignment(aln), model(m) {}

void Mcmc::burnin(int iterations, int tuneFrequency, int printFrequency){
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution unif(0.0,1.0);

    double currentLikelihood = model.lnLikelihood();
    double currentPrior = model.lnPrior();
    double currentPosterior = currentLikelihood + currentPrior;

    for(int i = 0; i < iterations; i++){
        double moveChoice = unif(generator);

        std::function<double()> proposal;
        int numGibbs = 5;

        if(moveChoice < 0.375){
            proposal = [this](){ return model.topologyMove();};
        }
        else if(moveChoice < 0.75){
            proposal = [this](){ return model.branchMove();};
        }
        else{
            proposal = [this](){ return model.stationaryMove();};
        }

        for(int j = 0; j < numGibbs; j++){
            double hastings = proposal();
            model.refreshLikelihood();

            double newPrior = model.lnPrior();
            double newLikelihood = model.lnLikelihood();
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

        if(i % printFrequency == 0){
            std::cout << i << "\t" << currentPosterior << "\tBranch Scale Rate:" << (double)model.acceptedBranchLength/(double)model.proposedBranchLength << 
                "\tNNI Rate:" << (double)model.acceptedNNI/(double)model.proposedNNI << "\tStationary Rate:" << (double)model.acceptedStationary/(double)model.proposedStationary <<
                "\tSubtree Scale Rate:" << (double)model.acceptedSubtreeScale/(double)model.proposedSubtreeScale << std::endl;
        }
        if(i % tuneFrequency == 0){
            model.tune();
        }
    }
}

void Mcmc::run(int iterations){

}