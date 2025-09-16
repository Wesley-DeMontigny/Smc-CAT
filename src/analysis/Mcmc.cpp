#include "Mcmc.hpp"
#include "core/Alignment.hpp"
#include "analysis/Model.hpp"
#include <random>
#include <cmath>

Mcmc::Mcmc(Alignment& aln, Model& m) : alignment(aln), model(m) {}

void Mcmc::burnin(int iterations){
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution unif(0.0,1.0);

    double currentPosterior = model.lnPrior() + model.lnLikelihood();

    for(int i = 0; i < iterations; i++){
        double hastings = model.treeMove();
        model.refreshLikelihood();

        double newPosterior = model.lnPrior() + model.lnLikelihood();
        double posteriorRatio = newPosterior - currentPosterior + hastings;
        if(std::log(unif(generator)) <= posteriorRatio){
            model.accept();
            currentPosterior = newPosterior;
        }
        else{
            model.reject();
        }
    }
}

void Mcmc::run(int iterations){

}