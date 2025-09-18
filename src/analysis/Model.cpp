#include "Model.hpp"
#include "Tree.hpp"
#include "core/Alignment.hpp"
#include "RateMatrices.hpp"
#include <random>
#include <iostream>
#include <cmath>

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

    auto generator = std::mt19937(std::random_device{}());
    std::uniform_real_distribution unifDist(0.0, 1.0);
    
    // Initialize the Chinese Restaurant Process
    for(int i = 0; i < numChar; i++){
        double randomVal = unifDist(generator);
        double total = dppAlpha/(i + dppAlpha);

        // If new category
        if(total > randomVal){
            auto newCat = TransitionProbabilityClass(numNodes, baseMatrix.get());
            newCat.members.insert(i);
            currentTransitionProbabilityClasses.push_back(newCat);
            continue;
        }

        // If old category
        for(auto &c : currentTransitionProbabilityClasses){
            total += c.members.size()/(i+dppAlpha);

            if(total > randomVal){
                c.members.insert(i);
                break;
            }
        }
    }

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
    acceptedSubtreeScale += (int)updateScaleSubtree;

    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedSubtreeScale += (int)updateScaleSubtree;

    updateStationary = false;
    updateNNI = false;
    updateBranchLength = false;
    updateScaleSubtree = false;
}

void Model::reject(){
    currentLnLikelihood = oldLnLikelihood;

    if(updateNNI || updateBranchLength || updateScaleSubtree){
        currentPhylogeny = oldPhylogeny;
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

    updateStationary = false;
    updateNNI = false;
    updateBranchLength = false;
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
    lnP += std::log(dppAlpha) * currentTransitionProbabilityClasses.size();

    for (auto& c : currentTransitionProbabilityClasses) {
        lnP += std::lgamma(c.members.size());
    }

    lnP += std::lgamma(dppAlpha) - std::lgamma(dppAlpha + numChar);

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
            auto pN = conditionaLikelihoodBuffer.get() + nIndex * numChar + (nWorkingSpace * numNodes * numChar);
            for(int i = 0; i < numChar; i++){
                pN[i] = Eigen::Vector<double, 20>::Ones();
            }

            for(auto d : n->descendants){
                int dIndex = d->id;
                uint8_t dWorkingSpace = currentConditionalLikelihoodFlags[dIndex];

                auto pD = conditionaLikelihoodBuffer.get() + dIndex * numChar + (dWorkingSpace * numNodes * numChar);

                Eigen::Matrix<double,20,1> workBuffer;

                /*
                    We are adopting a class-wise version of the inner loop. Indexing isn't as nice as incrementing a pointer
                    but when we have many classes we don't want to have to re-access a block of memory. Lets say that we have
                    100s of matrices - those aren't going to all fit in the cache. So if we alternate between matrix use due
                    to iterating over sites rather than classes, we risk additional cache misses
                */
                for(auto& tClass : currentTransitionProbabilityClasses){
                    Eigen::Matrix<double, 20, 20>& P = tClass.transitionProbabilities[dIndex];
                    for(int c : tClass.members){
                        workBuffer.noalias() = P * pD[c];
                        pN[c].array() *= workBuffer.array();
                    }
                }
            }
            double* rescalePointer = rescaleBuffer.get() + numChar * nIndex;
            std::fill(rescalePointer, rescalePointer + numChar, 0.0);

            for(int c = 0; c < numChar; c++){
                double max = pN->maxCoeff();
                *pN /= max;
                *rescalePointer = std::log(max);

                rescalePointer++;
                pN++;
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

    for(auto& tClass : currentTransitionProbabilityClasses){
        auto stationaryVec = (tClass.stationaryDistribution).array();
        for(int c : tClass.members){
            double siteLikelihood = (stationaryVec * (pR[c]).array()).sum();
            lnL += std::log(siteLikelihood);
        }
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

double Model::gibbsPartitionMove(){
    auto generator = std::mt19937(std::random_device{}());
    auto unifDist = std::uniform_real_distribution(0.0, 1.0);

    auto postOrder = currentPhylogeny.getPostOrder();

    int numAux = 5;
    int numIter = 25;

    double alphaSplit = std::log(dppAlpha/numAux);

    int bufferSize = (4 * numAux + currentTransitionProbabilityClasses.size());
    auto tempCLBuffer = new Eigen::Vector<double, 20>[numChar * bufferSize];
    auto tempRescaleBuffer = new double[numNodes * bufferSize];
    for(int i = 0; i < numNodes * bufferSize; i++){
        tempRescaleBuffer[i] = 0.0;
        tempCLBuffer[i] = Eigen::Vector<double, 20>::Zero();
    }

    // Choosing to update randomly seems like it is the best strategy?
    for(int iter = 0; iter < numIter; iter++){
        int randomSite = (int)(unifDist(generator) * numChar);
        int originalCategory = -1;
        for (int i = 0; i < currentTransitionProbabilityClasses.size(); i++) {
            if (currentTransitionProbabilityClasses[i].members.count(randomSite)) {
                originalCategory = i;
                break;
            }
        }

        // Unassign and delete the category if it is empty
        currentTransitionProbabilityClasses[originalCategory].members.erase(randomSite);
        if(currentTransitionProbabilityClasses[originalCategory].members.size() == 0){
            currentTransitionProbabilityClasses.erase(currentTransitionProbabilityClasses.begin() + originalCategory);
        }

        for(int n = 0; n < numAux; n++){
            auto auxCat = TransitionProbabilityClass(numNodes, baseMatrix.get());
            auxCat.recomputeEigens();
            for(auto n : postOrder){
                n->updateTP = false;
                auxCat.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
            }
            currentTransitionProbabilityClasses.push_back(auxCat);
        }

        int catSize = currentTransitionProbabilityClasses.size();

        // We try to give ourselves a big enough buffer at the beginning, but if it gets too small, update it.
        if(bufferSize < catSize){
            delete [] tempCLBuffer;
            delete [] tempRescaleBuffer;

            bufferSize = (4 * numAux + catSize);
            tempCLBuffer = new Eigen::Vector<double, 20>[numChar * bufferSize];
            tempRescaleBuffer = new double[numNodes * bufferSize];
            for(int i = 0; i < numNodes * bufferSize; i++){
                tempRescaleBuffer[i] = 0.0;
                tempCLBuffer[i] = Eigen::Vector<double, 20>::Zero();
            }
        }

        // Run the pruning algorithm
        for(auto n : postOrder){
            int nIndex = n->id;
            auto pN = tempCLBuffer + nIndex * bufferSize;
            if(!n->isTip){
                for(int i = 0; i < numChar; i++){
                    pN[i] = Eigen::Vector<double, 20>::Ones();
                }

                for(auto d : n->descendants){
                    int dIndex = d->id;

                    auto pD = tempCLBuffer+ dIndex * bufferSize;

                    Eigen::Matrix<double,20,1> workBuffer;

                    for(int c = 0; c < catSize; c++){
                        Eigen::Matrix<double, 20, 20>& P = currentTransitionProbabilityClasses[c].transitionProbabilities[dIndex];
                        workBuffer.noalias() = P * pD[c];
                        pN[c].array() *= workBuffer.array();
                    }
                }
                double* rescalePointer = tempRescaleBuffer + bufferSize * nIndex;
                std::fill(rescalePointer, rescalePointer + bufferSize, 0.0);

                for(int c = 0; c < catSize; c++){
                    double max = pN->maxCoeff();
                    *pN /= max;
                    *rescalePointer = std::log(max);

                    rescalePointer++;
                    pN++;
                }
            }
            else{
                for(int c = 0; c < catSize; c++){
                    pN[c] = *(conditionaLikelihoodBuffer.get() + nIndex * numChar + randomSite);
                }
            }
        }

        int rIndex = currentPhylogeny.getRoot()->id;
        auto pR = tempCLBuffer + rIndex * bufferSize;
        std::vector<double> likelihoodVec;

        for(int c = 0; c < catSize; c++){
            auto stationaryVec = (currentTransitionProbabilityClasses[c].stationaryDistribution).array();
            double siteLikelihood = (stationaryVec * (pR[c]).array()).sum();
            siteLikelihood = std::log(siteLikelihood);
            for(auto n : postOrder){
                siteLikelihood += (tempRescaleBuffer + bufferSize * n->id)[c];
            }
            if(c < catSize - numAux){
                likelihoodVec.push_back(siteLikelihood + std::log(currentTransitionProbabilityClasses[c].members.size()));
            }
            else {
                likelihoodVec.push_back(siteLikelihood + alphaSplit);
            }
        }

        double maxL = *std::max_element(likelihoodVec.begin(), likelihoodVec.end());
        double total = 0.0;
        for(double& d : likelihoodVec){
            d -= maxL;
            d = std::exp(d);
            total += d;
        }

        double categoryDraw = total * unifDist(generator);

        total = 0.0;
        bool assigned = false;
        for(int i = 0; i < likelihoodVec.size(); i++){
            total += likelihoodVec[i];
            if(total > categoryDraw){
                if(i < catSize - numAux) { //It already exists
                    for(int i = 0; i < numAux; i++)
                        currentTransitionProbabilityClasses.pop_back();
                    currentTransitionProbabilityClasses[i].members.insert(randomSite);
                }
                else {
                    // Delete the num aux other than the one we want.
                    for (int c = catSize - 1; c >= catSize - numAux; --c) {
                        if (c != i) {
                            currentTransitionProbabilityClasses.erase(currentTransitionProbabilityClasses.begin() + c);
                        }
                    }
                    currentTransitionProbabilityClasses[currentTransitionProbabilityClasses.size()-1].members.insert(randomSite);
                }
                break;
            }
        }
    }

    updateStationary = true;
    currentPhylogeny.updateAll();

    delete [] tempCLBuffer;
    delete [] tempRescaleBuffer;

    return INFINITY;
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