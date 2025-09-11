#include "Model.hpp"
#include "Tree.hpp"
#include "core/Alignment.hpp"
#include "RateMatrices.hpp"
#include "core/ProbabilityFunctions.hpp"
#include <random>

Model::Model(Alignment& aln) : 
                       numChar(aln.getNumChar()), currentPhylogeny(aln.getTaxaNames()),
                       oldPhylogeny(currentPhylogeny), numNodes(currentPhylogeny.getNumNodes()), numTaxa(aln.getNumTaxa()) {

    // We can make this configurable later
    baseMatrix = std::shared_ptr<Eigen::Matrix<double, 20, 20>>(new Eigen::Matrix<double, 20, 20>(RateMatrices::ConstructLG()));

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
    for(int i = 0; i < numChar * numNodes * 2; i++){
        rescaleBuffer[i] = 0.0;
        conditionaLikelihoodBuffer[i] = Eigen::Vector<double, 20>::Zero();
    }

    // Load the data into our vectors
    for(int i = 0; i < numChar; i++){
        for(int j = 0; j < numTaxa; j++){
            int code = aln(j, i);
            if(code < 20){
                conditionaLikelihoodBuffer[i + j*numChar][code] = 1.0;
                conditionaLikelihoodBuffer[i + j*numChar + numNodes*numChar][code] = 1.0;
            }
            else {
                conditionaLikelihoodBuffer[i + j*numChar] = Eigen::Vector<double, 20>::Ones(); // For ambiguous characters
                conditionaLikelihoodBuffer[i + j*numChar + numNodes*numChar] = Eigen::Vector<double, 20>::Ones();
            }
        }
    }

    currentPhylogeny.updateAll();

    currentTransitionProbabilityClasses.push_back(TransitionProbabilityClass(numChar, baseMatrix));

    for(auto& c : currentTransitionProbabilityClasses){
        c.recomputeEigens();
        for(auto n : currentPhylogeny.getPostOrder()){
            c.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
        }
    }

    oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;

    refreshLikelihood();
    accept();
}

void Model::accept(){
    if(updateLocal || updateNNI || updateBranchLength){
        oldPhylogeny = currentPhylogeny;
    }

    if(updateAssignments){
    std::copy(currentClassAssignments.get(),
        currentClassAssignments.get() + numChar,
        oldClassAssignments.get());
    }

    std::copy(rescaleBuffer.get(), 
            rescaleBuffer.get() + numChar * numNodes,
        rescaleBuffer.get() + numChar * numNodes);

    for(int i = 0; i < numChar * numNodes; i++){
        conditionaLikelihoodBuffer[numChar*numNodes + i] = conditionaLikelihoodBuffer[i];
    }
    
    if(updateLocal || updateBranchLength || updateStationary){
        oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;
    }

    acceptedLocal += (int)updateLocal;
    acceptedNNI += (int)updateNNI;
    acceptedBranchLength += (int)updateBranchLength;
    acceptedStationary += (int)updateStationary;
    acceptedAssignments += (int)updateAssignments;

    proposedLocal += (int)updateLocal;
    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedAssignments += (int)updateAssignments;

    bool updateLocal = false;
    bool updateStationary = false;
    bool updateNNI = false;
    bool updateBranchLength = false;
    bool updateAssignments = false;
}

void Model::reject(){
    if(updateLocal || updateNNI || updateBranchLength){
        currentPhylogeny = oldPhylogeny;
    }

    if(updateAssignments){
        std::copy(oldClassAssignments.get(),
            oldClassAssignments.get() + numChar,
            currentClassAssignments.get());
    }

    std::copy(rescaleBuffer.get() + numChar * numNodes,
        rescaleBuffer.get() + 2 * numChar * numNodes,
        rescaleBuffer.get());

    for(int i = 0; i < numChar * numNodes; i++){
        conditionaLikelihoodBuffer[i] = conditionaLikelihoodBuffer[numNodes*numChar + i];
    }

    if(updateLocal || updateBranchLength || updateStationary){
        currentTransitionProbabilityClasses = oldTransitionProbabilityClasses;
    }

    proposedLocal += (int)updateLocal;
    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedAssignments += (int)updateAssignments;

    bool updateLocal = false;
    bool updateStationary = false;
    bool updateNNI = false;
    bool updateBranchLength = false;
    bool updateAssignments = false;
}

double Model::lnLikelihood(){
    return currentLnLikelihood;
}

double Model::lnPrior(){
    double lnP = 0.0;

    double shape = (double)(numNodes - 1);
    double rate = 1.0;

    double lengthSum = 0.0;
    for(auto n : currentPhylogeny.getPostOrder()){
        lengthSum += n->branchLength;
    }

    lnP += ProbabilityFunctions::GammaLogPDF(lengthSum, shape, rate);

    // For now we will assume flat priors on the stationary

    return lnP;
}

void Model::refreshLikelihood(){
    auto postOrder = currentPhylogeny.getPostOrder();

    // All moves should be mutually exclusive each turn.
    if(updateStationary){ // Update all TPs of that class if that was the last move
        for(auto& c : currentTransitionProbabilityClasses){
            if(c.updated){
                c.recomputeEigens();
                for(auto n : postOrder){
                    n->updateTP = false;
                    c.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
                }
            }
        }
    }
    else if(updateLocal || updateBranchLength){ // Update specific branch lengths if that was the last move
        for(auto n : postOrder){
            if(n->updateTP){
                n->updateTP = false;
                for(auto c : currentTransitionProbabilityClasses){
                    c.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
                }
            }
        }
    }

    // Change the working space for everything that will need a CL update
    for(auto n : postOrder){
        if(n->updateCL){
            currentConditionalLikelihoodFlags[n->id] = (currentConditionalLikelihoodFlags[n->id] * -1) + 1;
        }
    }


    for(auto n : postOrder){
        int nIndex = n->id;
        uint8_t nWorkingSpace = currentConditionalLikelihoodFlags[nIndex];
        if(n->updateCL && !n->isTip){ // Tips never need their CL buffer updated because it just defined by the data
            auto pNN = conditionaLikelihoodBuffer.get() + nIndex * numChar + (nWorkingSpace * numNodes * numChar);
            for(int i = 0; i < numChar; i++){
                pNN[i] = Eigen::Vector<double, 20>::Ones();
            }

            for(auto d : n->descendants){
                int dIndex = d->id;
                uint8_t dWorkingSpace = currentConditionalLikelihoodFlags[dIndex];

                auto pN = pNN;
                auto pD = conditionaLikelihoodBuffer.get() + dIndex * numChar + (dWorkingSpace * numNodes * numChar);

                for(int c = 0; c < numChar; c++){
                    Eigen::Matrix<double, 20, 20> P = currentTransitionProbabilityClasses[currentClassAssignments[c]].transitionProbabilities[dIndex];
                    *pN = (*pN).array() * (P * (*pD)).array();
                    pD++;
                    pN++;
                }
            }
            double* rescalePointer = rescaleBuffer.get() + numChar * nIndex;
            std::fill(rescalePointer, rescalePointer + numChar, 0.0);

            for(int c = 0; c < numChar; c++){
                auto vec = *pNN;
                double max = -INFINITY;
                for(int i = 0; i < 20; i++){
                    if(vec[i] > max)
                        max = vec[i];
                }
                vec /= max;
                *rescalePointer = std::log(max);
                rescalePointer++;
                pNN++;
            }
        }
    }

    for(auto n : postOrder){
        n->updateCL = false;
    }


    int rIndex = currentPhylogeny.getRoot()->id;
    uint8_t dWorkingSpace = currentConditionalLikelihoodFlags[rIndex];
    auto pR = conditionaLikelihoodBuffer.get() + rIndex * numChar + (dWorkingSpace * numNodes * numChar);
    double lnL = 0.0;

    for(int c = 0; c < numChar; c++){
        auto stationaryVec = (currentTransitionProbabilityClasses[currentClassAssignments[c]].stationaryDistribution).array();
        double siteLikelihood = (stationaryVec * (*pR).array()).sum();
        lnL += std::log(siteLikelihood);
        pR ++;
    }

    double* rescalePointer = rescaleBuffer.get();
    for(int i = 0; i < numNodes * numChar; i++){
        lnL += *rescalePointer;
        rescalePointer++;
    }

    currentLnLikelihood = lnL;
}

double Model::treeMove(){

}

double Model::stationaryMove(){

}

void Model::tune(){
    
}