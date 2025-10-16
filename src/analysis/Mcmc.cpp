#include "Mcmc.hpp"
#include "analysis/Particle.hpp"
#include "core/RandomVariable.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <functional>

Mcmc::Mcmc(void) {}

double Mcmc::run(Particle& model, int iterations, const std::unordered_map<std::string, double>& splitPosterior, bool updateInvar, double tempering){
    auto& rng = RandomVariable::randomVariableInstance();

    model.refreshLikelihood();
    double currentLikelihood = model.lnLikelihood() * tempering;
    double currentPrior = model.lnPrior();
    double currentPosterior = currentLikelihood + currentPrior;

    bool sampleAlpha = model.getNumRates() > 1;

    for(int i = 0; i < iterations; i++){
        double moveChoice = rng.uniformRv();

        std::function<double(Particle&)> proposal;
        int numGibbs = 2;

        if(moveChoice < 0.2){
            proposal = [splitPosterior](Particle& m){ return m.topologyMove(splitPosterior); };
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

            if(std::log(rng.uniformRv()) <= posteriorRatio){
                model.accept();
                currentPosterior = newPosterior;
                currentPrior = newPrior;
                currentLikelihood = newLikelihood;
            }
            else{
                model.reject();
            }
        }
    }

    return currentPosterior;
}