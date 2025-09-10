#include "Model.hpp"
#include "Tree.hpp"
#include "core/Alignment.hpp"

Model::Model(Tree& t, Alignment& a, Eigen::Matrix<double, 20, 20> bM) : 
                       baseMatrix(bM), phylogeny(t), numChar(a.getNumChar()),
                       numNodes(t.getNumNodes()), numTaxa(t.getNumTaxa())  {
    
    currentConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes * numChar]);
    oldConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes * numChar]);
    for(int i = 0; i < numChar; i++){
        currentConditionalLikelihoodFlags[i] = 0;
        oldConditionalLikelihoodFlags[i] = 0;
    }

    currentClassAssignments = std::unique_ptr<int[]>(new int[numChar]);
    oldClassAssignments = std::unique_ptr<int[]>(new int[numChar]);
    for(int i = 0; i < numChar; i++){
        currentClassAssignments[i] = 0;
        oldClassAssignments[i] = 0;
    }

    conditionaLikelihoodBuffer = std::unique_ptr<Eigen::Vector<double, 20>[]>(new Eigen::Vector<double, 20>[numChar * numNodes * 2]);
    rescaleBuffer = std::unique_ptr<double[]>(new double[numChar * numNodes *2]);

    currentTransitionProbabilityClasses.push_back(TransitionProbabilityClass(numChar, baseMatrix));
    oldTransitionProbabilityClasses.push_back(TransitionProbabilityClass(numChar, baseMatrix));
}

void Model::accept(){

}

void Model::reject(){

}

double Model::lnLikelihood(){

}

double Model::lnPrior(){

}

void Model::refreshLikelihood(){

}

void Model::updateAssignments(int numUpdates){

}

double Model::updateStationary(){

}