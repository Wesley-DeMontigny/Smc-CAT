#include "Model.hpp"
#include "Tree.hpp"
#include "core/Alignment.hpp"
#include "RateMatrices.hpp"
#include <random>
#include <iostream>

double GammaLogPDF(double x, double shape, double rate){
    if (x <= 0.0) return -INFINITY; 
    return shape * std::log(rate) + (shape - 1.0) * std::log(x) - rate * x - std::lgamma(shape);
}


Model::Model(Alignment& aln) : 
                       numChar(aln.getNumChar()), currentPhylogeny(aln.getTaxaNames()),
                       oldPhylogeny(currentPhylogeny), numNodes(currentPhylogeny.getNumNodes()), numTaxa(aln.getNumTaxa()) {

    // We can make this configurable later
    baseMatrix = std::unique_ptr<Eigen::Matrix<double, 20, 20>>(new Eigen::Matrix<double, 20, 20>(RateMatrices::ConstructLG()));

    currentConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes]);
    oldConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes]);
    for(int i = 0; i < numNodes; i++){
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

    currentTransitionProbabilityClasses.push_back(TransitionProbabilityClass(numChar, baseMatrix.get()));

    for(auto& c : currentTransitionProbabilityClasses){
        c.recomputeEigens();
        for(auto n : currentPhylogeny.getPostOrder()){
            c.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
        }
    }

    currentPhylogeny.updateAll();
    refreshLikelihood();

    oldPhylogeny = currentPhylogeny;

    std::copy(rescaleBuffer.get(),
        rescaleBuffer.get() + numChar * numNodes,
        rescaleBuffer.get() + numChar * numNodes
    );

    std::copy(currentConditionalLikelihoodFlags.get(),
        currentConditionalLikelihoodFlags.get() + numNodes,
        oldConditionalLikelihoodFlags.get()
    );
    
    oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;

    oldLnLikelihood = currentLnLikelihood;
}

void Model::accept(){
    oldLnLikelihood = currentLnLikelihood;

    if(updateNNI || updateBranchLength){
        oldPhylogeny = currentPhylogeny;
    }

    if(updateAssignments){
        std::copy(currentClassAssignments.get(),
            currentClassAssignments.get() + numChar,
            oldClassAssignments.get()
        );
    }

    std::copy(rescaleBuffer.get(), 
        rescaleBuffer.get() + numChar * numNodes,
        rescaleBuffer.get() + numChar * numNodes
    );

    std::copy(currentConditionalLikelihoodFlags.get(), 
        currentConditionalLikelihoodFlags.get() + numNodes,
        oldConditionalLikelihoodFlags.get()
    );
    
    if(updateBranchLength || updateStationary){
        oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;
    }

    acceptedNNI += (int)updateNNI;
    acceptedBranchLength += (int)updateBranchLength;
    acceptedStationary += (int)updateStationary;
    acceptedAssignments += (int)updateAssignments;
    acceptedSubtreeScale += (int)updateScaleSubtree;

    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedAssignments += (int)updateAssignments;
    proposedSubtreeScale += (int)updateScaleSubtree;

    updateStationary = false;
    updateNNI = false;
    updateBranchLength = false;
    updateScaleSubtree = false;
    updateAssignments = false;
}

void Model::reject(){
    currentLnLikelihood = oldLnLikelihood;

    if(updateNNI || updateBranchLength || updateScaleSubtree){
        currentPhylogeny = oldPhylogeny;
    }

    if(updateAssignments){
        std::copy(oldClassAssignments.get(),
            oldClassAssignments.get() + numChar,
            currentClassAssignments.get()
        );
    }

    std::copy(rescaleBuffer.get() + numChar * numNodes,
        rescaleBuffer.get() + 2 * numChar * numNodes,
        rescaleBuffer.get()
    );

    std::copy(oldConditionalLikelihoodFlags.get(), 
        oldConditionalLikelihoodFlags.get() + numNodes,
        currentConditionalLikelihoodFlags.get()
    );

    if(updateBranchLength || updateStationary || updateScaleSubtree){
        currentTransitionProbabilityClasses = oldTransitionProbabilityClasses;
    }

    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedSubtreeScale += (int)updateScaleSubtree;
    proposedAssignments += (int)updateAssignments;

    updateStationary = false;
    updateNNI = false;
    updateBranchLength = false;
    updateAssignments = false;
    updateBranchLength = false;
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

    lnP += GammaLogPDF(lengthSum, shape, rate);

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
                c.updated = false;
            }
        }
    }
    else if(updateBranchLength || updateScaleSubtree){ // Update specific branch lengths if that was the last move
        for(auto n : postOrder){
            if(n->updateTP){
                n->updateTP = false;
                for(auto& c : currentTransitionProbabilityClasses){
                    c.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
                }
            }
        }
    }

    // Change the working space for everything that will need a CL update
    for(auto n : postOrder){
        if(n->updateCL){
            currentConditionalLikelihoodFlags[n->id] ^= 1;
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

                std::vector<Eigen::Matrix<double, 20, 20>> pMatrices;
                for(auto& c : currentTransitionProbabilityClasses){
                    pMatrices.push_back(c.transitionProbabilities[dIndex]);
                }

                for(int c = 0; c < numChar; c++){
                    Eigen::Matrix<double, 20, 20>& P = pMatrices[currentClassAssignments[c]];
                    *pN = (*pN).array() * (P * (*pD)).array();
                    pD++;
                    pN++;
                }
            }
            double* rescalePointer = rescaleBuffer.get() + numChar * nIndex;
            std::fill(rescalePointer, rescalePointer + numChar, 0.0);

            for(int c = 0; c < numChar; c++){
                double max = pNN->maxCoeff();
                *pNN /= max;
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

double Model::topologyMove(){
    auto generator = std::mt19937(std::random_device{}());

    double hastings = currentPhylogeny.NNIMove(generator);
    updateNNI = true;

    return hastings;
}

double Model::branchMove(){
    auto generator = std::mt19937(std::random_device{}());
    std::uniform_real_distribution unifDist(0.0, 1.0);

    double hastings = 0.0;

    if(unifDist(generator) < 0.75){
        hastings = currentPhylogeny.scaleBranchMove(scaleDelta, generator);
        updateBranchLength = true;
    }
    else {
        hastings = currentPhylogeny.scaleSubtreeMove(subtreeScaleDelta, generator);
        updateScaleSubtree = true;
    }

    return hastings;
}

double Model::stationaryMove(){
    auto generator = std::mt19937(std::random_device{}());
    auto unifDist = std::uniform_int_distribution<int>(0, currentTransitionProbabilityClasses.size() - 1);
    double randCategory = unifDist(generator);

    double hastings = currentTransitionProbabilityClasses[randCategory].dirichletSimplexMove(stationaryAlpha, generator);
    updateStationary = true;
    currentPhylogeny.updateAll();

    return hastings;
}

void Model::tune(){
    if(proposedBranchLength > 0){
        double blRate = (double)acceptedBranchLength/(double)proposedBranchLength;

        if ( blRate > 0.33 ) {
            scaleDelta *= (1.0 + ((blRate-0.33)/0.67));
        }
        else {
            scaleDelta /= (2.0 - blRate/0.33);
        }
        acceptedBranchLength = 0;
        proposedBranchLength = 0;
    }

    if(proposedSubtreeScale > 0){
        double ssRate = (double)acceptedSubtreeScale/(double)proposedSubtreeScale;

        if ( ssRate > 0.33 ) {
            subtreeScaleDelta *= (1.0 + ((ssRate-0.33)/0.67));
        }
        else {
            subtreeScaleDelta /= (2.0 - ssRate/0.33);
        }
        acceptedSubtreeScale = 0;
        proposedSubtreeScale = 0;
    }

    if(proposedStationary > 0){
        double stationaryRate = (double)acceptedStationary/(double)proposedStationary;

        if ( stationaryRate > 0.33 ) {
            stationaryAlpha /= (1.0 + ((stationaryRate-0.33)/0.67));
        }
        else {
            stationaryAlpha *= (2.0 - stationaryRate/0.33);
        }

        acceptedStationary = 0;
        proposedStationary = 0;
    }
}