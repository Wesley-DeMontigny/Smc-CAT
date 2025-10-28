#include "analysis/Particle.hpp"
#include "Mcmc.hpp"
#include <boost/random/uniform_01.hpp>
#include <cmath>
#include <functional>
#include <iostream>

Mcmc::Mcmc(void) : totalMoveWeight(0.0) {}

void Mcmc::emplaceMove(std::tuple<double, std::function<int(Particle&)>, std::function<double(Particle&)>>&& move){
    totalMoveWeight += std::get<0>(move);
    moves.emplace_back(move);
}

void Mcmc::initMoveProbs(){
    moveProbs.clear();

    double cumSum = 0.0;
    for(const auto& m : moves){
        cumSum += std::get<0>(m);
        moveProbs.push_back(cumSum / totalMoveWeight);
    }
}

double Mcmc::run(Particle& model, int iterations, double tempering){
    if(moveProbs.size() == 0){
        std::cout << "Error! The move probabilities are not initialized. Did you forget to call initMoveProbs?" << std::endl;
        std::exit(1);
    }

    boost::random::mt19937& rng = model.getRng();
    boost::random::uniform_01<double> unif{};

    model.refreshLikelihood();
    double currentLikelihood = model.lnLikelihood() * tempering;
    double currentPrior = model.lnPrior();
    double currentPosterior = currentLikelihood + currentPrior;

    for(int i = 0; i < iterations; i++){
        double moveChoice = unif(rng);

        std::function<double(Particle&)> proposal;
        int numGibbs = 3;
        for(int j = 0; j < moveProbs.size(); j++){
            if(moveChoice < moveProbs[j]){
                proposal = std::get<2>(moves[j]);
                numGibbs = std::get<1>(moves[j])(model);
                break;
            }
        }


        for(int j = 0; j < numGibbs; j++){
            double hastings = proposal(model);
            model.refreshLikelihood();

            double newPrior = model.lnPrior();
            double newLikelihood = model.lnLikelihood() * tempering;
            double newPosterior = newPrior + newLikelihood;
            double posteriorRatio = newPosterior - currentPosterior + hastings;

            if(std::log(unif(rng)) <= posteriorRatio){
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