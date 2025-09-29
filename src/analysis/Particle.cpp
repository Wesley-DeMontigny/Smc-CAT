#include "Particle.hpp"
#include "Tree.hpp"
#include "core/Alignment.hpp"
#include "RateMatrices.hpp"
#include "H5Cpp.h"
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <iostream>
#include <cmath>
#include <filesystem>

#if TIME_PROFILE == 1
    #include <chrono>
#endif

double gammaLogPDF(double x, double shape, double rate){
    if (x <= 0.0) return -INFINITY; 
    return shape * std::log(rate) + (shape - 1.0) * std::log(x) - rate * x - std::lgamma(shape);
}

double sampleBeta(double alpha, double beta, std::mt19937& gen){
    auto alphaDist = std::gamma_distribution(alpha, 1.0);
    auto betaDist = std::gamma_distribution(beta, 1.0);

    double x = alphaDist(gen);
    double y = betaDist(gen);

    return x / (x + y);
}

double betaLogPDF(double x, double alpha, double beta){
    if(x >= 1 || x <= 0) return -INFINITY;

    return std::pow(x, alpha - 1.0) * std::pow(1.0 - x, beta - 1.0);
}

Particle::Particle(Alignment& aln, bool fullInitialization) : 
                       numChar(aln.getNumChar()), currentPhylogeny(aln.getTaxaNames()), aln(aln),
                       oldPhylogeny(currentPhylogeny), numNodes(currentPhylogeny.getNumNodes()), numTaxa(aln.getNumTaxa()) {

    // We can make this configurable later
    baseMatrix = std::unique_ptr<Eigen::Matrix<double, 20, 20>>(new Eigen::Matrix<double, 20, 20>(RateMatrices::ConstructLG()));

    currentConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes]);
    oldConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes]);
    conditionaLikelihoodBuffer = std::unique_ptr<Eigen::Vector<double, 20>[]>(new Eigen::Vector<double, 20>[numChar * numNodes * 2]);
    rescaleBuffer = std::unique_ptr<double[]>(new double[numChar * numNodes *2]);
    for(int i = 0; i < numChar * numNodes * 2; i++){
        rescaleBuffer[i] = 0.0;
        conditionaLikelihoodBuffer[i] = Eigen::Vector<double, 20>::Zero();
    }

    // Load the data into our vectors
    isInvariant = std::unique_ptr<double[]>(new double[numChar]);
    invariantCharacter = std::unique_ptr<int[]>(new int[numChar]);
    for(int i = 0; i < numChar; i++){
        int code1 = aln(0, i);
        double invariant = 1.0;
        for(int j = 0; j < numTaxa; j++){
            int code = aln(j, i);
            if(code != code1)
                invariant = 0.0;
            
            if(code < 20){
                conditionaLikelihoodBuffer[i + j*numChar][code] = 1.0;
                conditionaLikelihoodBuffer[i + j*numChar + numNodes*numChar][code] = 1.0;
            }
            else {
                conditionaLikelihoodBuffer[i + j*numChar] = Eigen::Vector<double, 20>::Ones(); // For ambiguous characters
                conditionaLikelihoodBuffer[i + j*numChar + numNodes*numChar] = Eigen::Vector<double, 20>::Ones();
                invariant = 0.0;
            }
        }
        isInvariant[i] = invariant;

        if(invariant == 1.0){
            invariantCharacter[i] = code1;
        }
        else{
            invariantCharacter[i] = 0;
        }
    }

    initialize(fullInitialization);
}

void Particle::initialize(bool full){
    // We waste a little bit of compute the first time we initilize this particle object, but this is easiest right now
    currentPhylogeny = Tree(aln.getTaxaNames());

    for(int i = 0; i < numNodes; i++){
        currentConditionalLikelihoodFlags[i] = 0;
        oldConditionalLikelihoodFlags[i] = 0;
    }

    currentTransitionProbabilityClasses.clear();

    auto generator = std::mt19937(std::random_device{}());
    std::uniform_real_distribution unifDist(0.0, 1.0);
    
    // I am just going to assume a uniform prior for now
    currentPInvar = unifDist(generator);
    oldPInvar = currentPInvar;

    // Only do these computations if we are initializing with the full dataset
    if(full){
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
    }

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

void Particle::accept(){
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

    oldPInvar = currentPInvar;

    acceptedNNI += (int)updateNNI;
    acceptedBranchLength += (int)updateBranchLength;
    acceptedStationary += (int)updateStationary;
    acceptedSubtreeScale += (int)updateScaleSubtree;
    acceptedInvar += (int)updateInvar;

    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedSubtreeScale += (int)updateScaleSubtree;
    proposedInvar += (int)updateInvar;

    updateStationary = false;
    updateNNI = false;
    updateBranchLength = false;
    updateScaleSubtree = false;
    updateInvar = false;
}

void Particle::reject(){
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

    currentPInvar = oldPInvar;

    proposedNNI += (int)updateNNI;
    proposedBranchLength += (int)updateBranchLength;
    proposedStationary += (int)updateStationary;
    proposedSubtreeScale += (int)updateScaleSubtree;
    proposedInvar += (int)updateInvar;

    updateStationary = false;
    updateNNI = false;
    updateBranchLength = false;
    updateBranchLength = false;
    updateInvar = false;
}

double Particle::lnLikelihood(){
    return currentLnLikelihood;
}

double Particle::lnPrior(){
    double lnP = 0.0;

    double shape = (double)(numNodes - 1);
    double rate = 10.0;

    double lengthSum = 0.0;
    for(auto n : currentPhylogeny.getPostOrder()){
        lengthSum += n->branchLength;
    }

    lnP += gammaLogPDF(lengthSum, shape, rate);

    // For now we will assume flat priors on the stationary
    lnP += std::log(dppAlpha) * currentTransitionProbabilityClasses.size();

    int totalActive = 0;
    for (auto& c : currentTransitionProbabilityClasses) {
        int memberCount = c.members.size();
        lnP += std::lgamma(memberCount);
        totalActive += memberCount;
    }

    lnP += std::lgamma(dppAlpha) - std::lgamma(dppAlpha + totalActive);

    return lnP;
}

void Particle::refreshLikelihood(){
    auto postOrder = currentPhylogeny.getPostOrder();

    int activeSites = 0;
    for(auto& c : currentTransitionProbabilityClasses){
        activeSites += c.members.size();
    }

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point preTPUpdate = std::chrono::steady_clock::now();
    #endif

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

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point postTPUpdate = std::chrono::steady_clock::now();
    std::cout << "The transition probability update completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(postTPUpdate - preTPUpdate).count() << "[milliseconds]" << std::endl;
    #endif

    // Change the working space for everything that will need a CL update
    for(auto n : postOrder){
        if(n->updateCL){
            currentConditionalLikelihoodFlags[n->id] ^= 1;
        }
    }

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point prePrune = std::chrono::steady_clock::now();
    #endif

    for(auto n : postOrder){
        int nIndex = n->id;
        uint8_t nWorkingSpace = currentConditionalLikelihoodFlags[nIndex];
        if(n->updateCL && !n->isTip){ // Tips never need their CL buffer updated because it just defined by the data
            auto pN = conditionaLikelihoodBuffer.get() + nIndex * numChar + (nWorkingSpace * numNodes * numChar);
            for(int i = 0; i < activeSites; i++){
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
            std::fill(rescalePointer, rescalePointer + activeSites, 0.0);

            for(int c = 0; c < activeSites; c++){
                double max = pN->maxCoeff();
                *pN /= max;
                *rescalePointer = std::log(max);

                rescalePointer++;
                pN++;
            }
        }
    }

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point postPrune = std::chrono::steady_clock::now();
    std::cout << "Felsenstein's pruning algorithm completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(postPrune - prePrune).count() << "[milliseconds]" << std::endl;
    #endif

    for(auto n : postOrder){
        n->updateCL = false;
    }


    int rIndex = currentPhylogeny.getRoot()->id;
    uint8_t dWorkingSpace = currentConditionalLikelihoodFlags[rIndex];
    auto pR = conditionaLikelihoodBuffer.get() + rIndex * numChar + (dWorkingSpace * numNodes * numChar);
    double lnL = 0.0;

    double invInvarScaler = 1.0 - currentPInvar;

    for(auto& tClass : currentTransitionProbabilityClasses){
        auto stationaryVec = (tClass.stationaryDistribution).array();
        for(int c : tClass.members){
            double siteLikelihood = (stationaryVec * (pR[c]).array()).sum();

            lnL += std::log(siteLikelihood * invInvarScaler + currentPInvar * stationaryVec[invariantCharacter[c]] * isInvariant[c]);
        }
    }

    // It is hard to make this respect only active sites since it is ordered by node
    double* rescalePointer = rescaleBuffer.get();
    for(int i = 0; i < numNodes * numChar; i++){
        lnL += *rescalePointer;
        rescalePointer++;
    }

    currentLnLikelihood = lnL;
}

double Particle::computeSiteLikelihood(double tempering, int site){
    auto generator = std::mt19937(std::random_device{}());
    auto unifDist = std::uniform_real_distribution(0.0, 1.0);

    auto postOrder = currentPhylogeny.getPostOrder();

    // We are assuming the site has already been assigned and that the transition probability has been properly updated
    int siteAssignment = -1;
    for (int i = 0; i < currentTransitionProbabilityClasses.size(); i++) {
        if (currentTransitionProbabilityClasses[i].members.count(site)) {
            siteAssignment = i;
            break;
        }
    }
    assert(siteAssignment >= 0);

    int bufferSize = 1; // Until we include gamma site rate heterogeneity
    auto tempCLBuffer = new Eigen::Vector<double, 20>[numNodes * bufferSize];
    auto tempRescaleBuffer = new double[numNodes * bufferSize];
    for(int i = 0; i < numNodes * bufferSize; i++){
        tempRescaleBuffer[i] = 0.0;
        tempCLBuffer[i] = Eigen::Vector<double, 20>::Zero();
    }

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point preIter = std::chrono::steady_clock::now();
    #endif

    // Run the pruning algorithm
    for(auto n : postOrder){
        int nIndex = n->id;
        auto pN = tempCLBuffer + nIndex * bufferSize;
        if(!n->isTip){
            for(int i = 0; i < bufferSize; i++){
                pN[i] = Eigen::Vector<double, 20>::Ones();
            }

            for(auto d : n->descendants){
                int dIndex = d->id;

                auto pD = tempCLBuffer+ dIndex * bufferSize;

                Eigen::Matrix<double,20,1> workBuffer;

                Eigen::Matrix<double, 20, 20>& P = currentTransitionProbabilityClasses[siteAssignment].transitionProbabilities[dIndex];
                workBuffer.noalias() = P * *pD;
                (*pN).array() *= workBuffer.array();
            }
            double* rescalePointer = tempRescaleBuffer + bufferSize * nIndex;
            std::fill(rescalePointer, rescalePointer + bufferSize, 0.0);

            double max = pN->maxCoeff();
            *pN /= max;
            *rescalePointer = std::log(max);
        }
        else{
            *pN = *(conditionaLikelihoodBuffer.get() + nIndex * numChar + site);
        }
    }

    int rIndex = currentPhylogeny.getRoot()->id;
    auto pR = tempCLBuffer + rIndex * bufferSize;
    std::vector<double> likelihoodVec;

    double invInvarScaler = 1.0 - currentPInvar;

    auto stationaryVec = (currentTransitionProbabilityClasses[siteAssignment].stationaryDistribution).array();
    double siteLikelihood = (stationaryVec * (*pR).array()).sum();
    siteLikelihood = std::log(siteLikelihood * invInvarScaler + currentPInvar * stationaryVec[invariantCharacter[siteAssignment]] * isInvariant[siteAssignment]);
    for(int i = 0; i < numNodes * bufferSize; i++){
        siteLikelihood += tempRescaleBuffer[i];
    }
    siteLikelihood *= tempering;

    delete [] tempCLBuffer;
    delete [] tempRescaleBuffer;

    return siteLikelihood;
}

double Particle::topologyMove(){
    auto generator = std::mt19937(std::random_device{}());

    double hastings = currentPhylogeny.NNIMove(generator);
    updateNNI = true;

    return hastings;
}

double Particle::branchMove(){
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

double Particle::stationaryMove(){
    auto generator = std::mt19937(std::random_device{}());
    auto unifDist = std::uniform_int_distribution<int>(0, currentTransitionProbabilityClasses.size() - 1);
    double randCategory = unifDist(generator);

    double hastings = currentTransitionProbabilityClasses[randCategory].dirichletSimplexMove(stationaryAlpha, generator);
    updateStationary = true;
    currentPhylogeny.updateAll();

    return hastings;
}

double Particle::invarMove(){
    auto generator = std::mt19937(std::random_device{}());

    double a = invarAlpha + 1.0;
    double b = (invarAlpha / currentPInvar) - a + 2.0;
    double newPInvar = sampleBeta(a, b, generator);

    double a2 = invarAlpha + 1.0;
    double b2 = (invarAlpha / newPInvar) - a2 + 2.0;

    double forward = betaLogPDF(newPInvar, a, b);
    double backward = betaLogPDF(currentPInvar, a2, b2);

    currentPInvar = newPInvar;

    updateInvar = true;

    return backward - forward;
}

double Particle::gibbsPartitionMove(double tempering, int range){
    auto generator = std::mt19937(std::random_device{}());
    auto unifDist = std::uniform_real_distribution(0.0, 1.0);

    auto postOrder = currentPhylogeny.getPostOrder();

    int numAux = 5;
    int numIter = range;

    double alphaSplit = std::log(dppAlpha/numAux);

    int bufferSize = (4 * numAux + currentTransitionProbabilityClasses.size());
    auto tempCLBuffer = new Eigen::Vector<double, 20>[numNodes * bufferSize];
    auto tempRescaleBuffer = new double[numNodes * bufferSize];
    for(int i = 0; i < numNodes * bufferSize; i++){
        tempRescaleBuffer[i] = 0.0;
        tempCLBuffer[i] = Eigen::Vector<double, 20>::Zero();
    }

    // Choosing to update randomly seems like it is the best strategy?
    for(int iter = 0; iter < numIter; iter++){
        #if TIME_PROFILE==1
        std::chrono::steady_clock::time_point preIter = std::chrono::steady_clock::now();
        #endif
        int randomSite = (int)(unifDist(generator) * range);
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
            tempCLBuffer = new Eigen::Vector<double, 20>[numNodes * bufferSize];
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
                for(int i = 0; i < bufferSize; i++){
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

        double invInvarScaler = 1.0 - currentPInvar;

        for(int c = 0; c < catSize; c++){
            auto stationaryVec = (currentTransitionProbabilityClasses[c].stationaryDistribution).array();
            double siteLikelihood = (stationaryVec * (pR[c]).array()).sum();
            siteLikelihood = std::log(siteLikelihood * invInvarScaler + currentPInvar * stationaryVec[invariantCharacter[c]] * isInvariant[c]);
            for(auto n : postOrder){
                siteLikelihood += (tempRescaleBuffer + bufferSize * n->id)[c];
            }
            siteLikelihood *= tempering;
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
        for(int i = 0; i < likelihoodVec.size(); i++){
            total += likelihoodVec[i];
            if(total > categoryDraw){
                if(i < catSize - numAux) { //It already exists
                    for(int j = 0; j < numAux; j++)
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

        #if TIME_PROFILE==1
        std::chrono::steady_clock::time_point postIter = std::chrono::steady_clock::now();
        std::cout << "CRP Gibbs iteration completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(postIter - preIter).count() << "[milliseconds]" << std::endl;
        #endif
    }

    updateStationary = true;
    currentPhylogeny.updateAll();

    delete [] tempCLBuffer;
    delete [] tempRescaleBuffer;

    return INFINITY;
}

double Particle::assignSite(double tempering, int site){
    auto generator = std::mt19937(std::random_device{}());
    auto unifDist = std::uniform_real_distribution(0.0, 1.0);

    auto postOrder = currentPhylogeny.getPostOrder();

    int numAux = 5;

    double alphaSplit = std::log(dppAlpha/numAux);

    int bufferSize = (4 * numAux + currentTransitionProbabilityClasses.size());
    auto tempCLBuffer = new Eigen::Vector<double, 20>[numNodes * bufferSize];
    auto tempRescaleBuffer = new double[numNodes * bufferSize];
    for(int i = 0; i < numNodes * bufferSize; i++){
        tempRescaleBuffer[i] = 0.0;
        tempCLBuffer[i] = Eigen::Vector<double, 20>::Zero();
    }

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point preIter = std::chrono::steady_clock::now();
    #endif

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
        tempCLBuffer = new Eigen::Vector<double, 20>[numNodes * bufferSize];
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
            for(int i = 0; i < bufferSize; i++){
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
                pN[c] = *(conditionaLikelihoodBuffer.get() + nIndex * numChar + site);
            }
        }
    }

    int rIndex = currentPhylogeny.getRoot()->id;
    auto pR = tempCLBuffer + rIndex * bufferSize;
    std::vector<double> likelihoodVec;

    double invInvarScaler = 1.0 - currentPInvar;

    for(int c = 0; c < catSize; c++){
        auto stationaryVec = (currentTransitionProbabilityClasses[c].stationaryDistribution).array();
        double siteLikelihood = (stationaryVec * (pR[c]).array()).sum();
        siteLikelihood = std::log(siteLikelihood * invInvarScaler + currentPInvar * stationaryVec[invariantCharacter[c]] * isInvariant[c]);
        for(auto n : postOrder){
            siteLikelihood += (tempRescaleBuffer + bufferSize * n->id)[c];
        }
        siteLikelihood *= tempering;
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
    int assignment = 0;
    for(int i = 0; i < likelihoodVec.size(); i++){
        total += likelihoodVec[i];
        if(total > categoryDraw){
            assignment = i;
            if(i < catSize - numAux) { //It already exists
                for(int j = 0; j < numAux; j++)
                    currentTransitionProbabilityClasses.pop_back();
                currentTransitionProbabilityClasses[i].members.insert(site);
            }
            else {
                // Delete the num aux other than the one we want.
                for (int c = catSize - 1; c >= catSize - numAux; --c) {
                    if (c != i) {
                        currentTransitionProbabilityClasses.erase(currentTransitionProbabilityClasses.begin() + c);
                    }
                }
                currentTransitionProbabilityClasses[currentTransitionProbabilityClasses.size()-1].members.insert(site);
            }
            break;
        }
    }

    delete [] tempCLBuffer;
    delete [] tempRescaleBuffer;

    return likelihoodVec[assignment];
}

void Particle::tune(){
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

    if(proposedInvar > 0){
        double invarRate = (double)acceptedInvar/(double)proposedInvar;

        if ( invarRate > 0.33 ) {
            invarAlpha /= (1.0 + ((invarRate-0.33)/0.67));
        }
        else {
            invarAlpha *= (2.0 - invarRate/0.33);
        }

        acceptedInvar = 0;
        proposedInvar = 0;
    }
}

void Particle::write(const int id, const std::string& dir){
    using namespace H5;

    std::vector<int> assignmentVector(numChar, -1);
    for(int i = 0; i < currentTransitionProbabilityClasses.size(); i++){
        auto cls = currentTransitionProbabilityClasses[i];
        for(int m : cls.members)
            assignmentVector[m] = i;
    }


    std::ostringstream groupName;
    groupName << "state_" << id;

    H5File h5File;

    if (std::filesystem::exists(dir)) {
        h5File = H5File(dir, H5F_ACC_RDWR);
    } else {
        h5File = H5File(dir, H5F_ACC_TRUNC);
    }

    // Create a group for this particle
    Group group = h5File.createGroup("/" + groupName.str());

    // Write Newick
    StrType strDataType(PredType::C_S1, H5T_VARIABLE);
    DataSpace strDataSpace(H5S_SCALAR);
    DataSet newickData = group.createDataSet("tree", strDataType, strDataSpace);
    std::string newickStr = currentPhylogeny.generateNewick();
    newickData.write(&newickStr, strDataType);

    // Write the assignment matrix
    hsize_t assignmentDims[1] = { assignmentVector.size() };
    DataSpace assignmentSpace(1, assignmentDims);
    DataSet assignmentData = group.createDataSet("assignments", PredType::NATIVE_INT32, assignmentSpace);
    assignmentData.write(assignmentVector.data(), PredType::NATIVE_INT32);

    DataSpace scalarSpace(H5S_SCALAR);
    DataSet pInvarDataset = group.createDataSet("pInvar", PredType::NATIVE_DOUBLE, scalarSpace);
    pInvarDataset.write(&currentPInvar, PredType::NATIVE_DOUBLE);

    // Write out stationaries as a flattened 2D matrix padded with NaNs
    size_t maxLen = numChar;
    std::vector<double> buffer(20 * numChar, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < currentTransitionProbabilityClasses.size(); i++) {
        for (int j = 0; j < 20; j++) {
            buffer[i * 20 + j] = currentTransitionProbabilityClasses[i].stationaryDistribution[j];
        }
    }

    hsize_t dims[2] = { 20, maxLen };
    DataSpace stationarySpace(2, dims);
    DataSet stationaryData = group.createDataSet("stationaries", PredType::NATIVE_DOUBLE, stationarySpace);
    stationaryData.write(buffer.data(), PredType::NATIVE_DOUBLE);
}

void Particle::read(const int id, const std::string& dir){
    using namespace H5;
    std::ostringstream groupName;
    groupName << "state_" << id;

    H5File h5File(dir, H5F_ACC_RDONLY);
    Group group = h5File.openGroup("/" + groupName.str());

    // Read Newick
    DataSet newickData = group.openDataSet("tree");
    StrType strDataType(PredType::C_S1, H5T_VARIABLE);
    H5std_string newick;
    newickData.read(newick, strDataType);
    currentPhylogeny = Tree(newick, aln.getTaxaNames());

    // Read assignments
    DataSet assignmentData = group.openDataSet("assignments");
    DataSpace assignmentSpace = assignmentData.getSpace();

    DataSet pInvarData = group.openDataSet("pInvar");
    pInvarData.read(&currentPInvar, PredType::NATIVE_DOUBLE);

    hsize_t assignmentDims[1];
    assignmentSpace.getSimpleExtentDims(assignmentDims, nullptr);
    std::vector<int> assignmentVector(assignmentDims[0]);
    assignmentData.read(assignmentVector.data(), PredType::NATIVE_INT32);

    currentTransitionProbabilityClasses.clear();

    // Read stationaries
    DataSet stationaryData = group.openDataSet("stationaries");
    DataSpace stationarySpace = stationaryData.getSpace();

    hsize_t dims[2];
    stationarySpace.getSimpleExtentDims(dims, nullptr);
    size_t rows = dims[0];
    size_t cols = dims[1];

    std::vector<double> buffer(rows * cols);
    stationaryData.read(buffer.data(), PredType::NATIVE_DOUBLE);

    int maxClass = *std::max_element(assignmentVector.begin(), assignmentVector.end());

    for (int i = 0; i <= maxClass; i++) {
        TransitionProbabilityClass cls(numChar, baseMatrix.get());

        for (size_t j = 0; j < rows; j++) {
            double val = buffer[i * rows + j];
            if (!std::isnan(val)) {
                cls.stationaryDistribution[j] = val;
            }
        }

        for (size_t m = 0; m < assignmentVector.size(); m++) {
            if (assignmentVector[m] == i) {
                cls.members.insert(m);
            }
        }

        cls.recomputeEigens();
        for(auto n : currentPhylogeny.getPostOrder()){
            cls.recomputeTransitionProbs(n->id, n->branchLength, 1.0);
        }

        currentTransitionProbabilityClasses.push_back(cls);
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
    
    oldPInvar = currentPInvar;

    oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;

    oldLnLikelihood = currentLnLikelihood;
}