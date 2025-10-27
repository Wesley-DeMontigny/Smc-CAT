#include "core/Alignment.hpp"
#include "Particle.hpp"
#include "RateMatrices.hpp"
#include "SerializedParticle.hpp"
#include "Tree.hpp"
#include <boost/math/distributions.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if TIME_PROFILE == 1
    #include <chrono>
#endif

Particle::Particle(int seed, Alignment& aln, int nR, bool initInvar) : 
                       rng(seed), numChar(aln.getNumChar()), currentPhylogeny(rng, aln.getTaxaNames()), aln(aln), numRates(nR),
                       oldPhylogeny(currentPhylogeny), numNodes(aln.getNumTaxa() * 2 - 1), numTaxa(aln.getNumTaxa()) {

    // We can make this configurable later
    baseMatrix = std::unique_ptr<Eigen::Matrix<double, 20, 20>>(new Eigen::Matrix<double, 20, 20>(RateMatrices::ConstructLG()));
    
    // Reserve the spots for the rates. In initialization we can do gamma site rate heterogeneity if the numRates > 1
    currentRates.reserve(numRates);
    for (int r = 0; r < numRates; r++) {
        currentRates.emplace_back(1.0);
    }
    oldRates = currentRates;

    currentConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes]);
    oldConditionalLikelihoodFlags = std::unique_ptr<uint8_t[]>(new uint8_t[numNodes]);
    conditionaLikelihoodBuffer = std::unique_ptr<Eigen::Vector<CL_TYPE, 20>[]>(new Eigen::Vector<CL_TYPE, 20>[numChar * numNodes * numRates * 2]);
    rescaleBuffer = std::unique_ptr<double[]>(new double[numChar * numNodes * numRates * 2]);
    for(int i = 0; i < numChar * numNodes * numRates * 2; i++){
        rescaleBuffer[i] = 0.0;
        conditionaLikelihoodBuffer[i] = Eigen::Vector<CL_TYPE, 20>::Zero();
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
                for(int k = 0; k < numRates; k++){
                    conditionaLikelihoodBuffer[k + i*numRates + j*numChar*numRates][code] = 1.0;
                    conditionaLikelihoodBuffer[k + i*numRates + j*numChar*numRates + numNodes*numChar*numRates][code] = 1.0;
                }
            }
            else {
                for(int k = 0; k < numRates; k++){
                    conditionaLikelihoodBuffer[k + i*numRates + j*numChar*numRates].setOnes();// For ambiguous characters
                    conditionaLikelihoodBuffer[k + i*numRates + j*numChar*numRates + numNodes*numChar*numRates].setOnes();
                }
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

    initialize(initInvar);
}

void Particle::initialize(bool initInvar){
    // We waste a little bit of compute the first time we initilize this particle object, but this is easiest right now
    currentPhylogeny = Tree(rng, aln.getTaxaNames());

    for(int i = 0; i < numNodes; i++){
        currentConditionalLikelihoodFlags[i] = 0;
        oldConditionalLikelihoodFlags[i] = 0;
    }

    currentTransitionProbabilityClasses.clear();

    boost::random::uniform_01 unif{};


    // I am just going to assume a uniform prior for now
    if(initInvar){
        currentPInvar = unif(rng);
        oldPInvar = currentPInvar;
    }

    if(numRates > 1){
        currentShape = boost::random::gamma_distribution(3.0, 0.5)(rng); 
        oldShape = currentShape;
        discretizeGamma(currentRates, currentShape, numRates);
        oldRates = currentRates;
    }

    // Initialize the Chinese Restaurant Process
    for(int i = 0; i < numChar; i++){
        double randomVal = unif(rng);
        double total = dppAlpha/(i + dppAlpha);

        // If new category
        if(total > randomVal){
            auto newCat = TransitionProbabilityClass(rng, numNodes, numRates, baseMatrix.get());
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

    refreshLikelihood(true);

    oldPhylogeny = currentPhylogeny;

    std::copy(rescaleBuffer.get(),
        rescaleBuffer.get() + numChar * numNodes * numRates,
        rescaleBuffer.get() + numChar * numNodes * numRates
    );

    std::copy(currentConditionalLikelihoodFlags.get(),
        currentConditionalLikelihoodFlags.get() + numNodes,
        oldConditionalLikelihoodFlags.get()
    );
    
    oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;

    oldLnLikelihood = currentLnLikelihood;
}

void Particle::copy(Particle& p){
    *baseMatrix = *p.baseMatrix;

    currentTransitionProbabilityClasses = p.currentTransitionProbabilityClasses;
    oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;

    currentPhylogeny = p.currentPhylogeny;
    oldPhylogeny = currentPhylogeny;

    currentPInvar = p.currentPInvar;
    oldPInvar = currentPInvar;

    currentRates = p.currentRates;
    oldRates = currentRates;
    currentShape = p.currentShape;
    oldShape = currentShape;

    refreshLikelihood(true);

    oldPhylogeny = currentPhylogeny;

    std::copy(rescaleBuffer.get(),
        rescaleBuffer.get() + numChar * numNodes * numRates,
        rescaleBuffer.get() + numChar * numNodes * numRates
    );

    std::copy(currentConditionalLikelihoodFlags.get(),
        currentConditionalLikelihoodFlags.get() + numNodes,
        oldConditionalLikelihoodFlags.get()
    );
}

void Particle::copyFromSerialized(SerializedParticle& sp){
    currentPhylogeny = Tree(sp.newick, aln.getTaxaNames());
    currentShape = sp.shape;
    currentPInvar = sp.pInvar;
    
    if(numRates > 1)
        discretizeGamma(currentRates, currentShape, numRates);

    const std::vector<TreeNode*>& nodes = currentPhylogeny.getPostOrder();

    #if TIME_PROFILE == 1
    std::chrono::steady_clock::time_point preTransProb = std::chrono::steady_clock::now();
    #endif

    int oldSize = currentTransitionProbabilityClasses.size();
    int targetSize = sp.stationaries.size();
    if(oldSize > targetSize)
        currentTransitionProbabilityClasses.erase(
            currentTransitionProbabilityClasses.begin() + targetSize,
            currentTransitionProbabilityClasses.end()
        );

    for (int i = 0; i < targetSize; i++) {
        if(i < oldSize){
            TransitionProbabilityClass& cls = currentTransitionProbabilityClasses[i];

            cls.stationaryDistribution = sp.stationaries[i];
            *cls.baseMatrix = sp.baseMatrix;

            cls.members.clear();
            for (size_t m = 0; m < sp.assignments.size(); m++){
                if (sp.assignments[m] == i){
                    cls.members.insert(m);
                }
            }
        }
        else{
            TransitionProbabilityClass cls(rng, numNodes, numRates, baseMatrix.get());

            cls.stationaryDistribution = sp.stationaries[i];
            *cls.baseMatrix = sp.baseMatrix;

            for (size_t m = 0; m < sp.assignments.size(); m++){
                if (sp.assignments[m] == i){
                    cls.members.insert(m);
                }
            }

            currentTransitionProbabilityClasses.push_back(cls);
        }
    }

    #if TIME_PROFILE == 1
    std::chrono::steady_clock::time_point postTransProb = std::chrono::steady_clock::now();
    std::cout << "The transition prob copy completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(postTransProb - preTransProb).count() << "[milliseconds]" << std::endl;
    #endif

    refreshLikelihood(true);

    std::copy(rescaleBuffer.get(),
        rescaleBuffer.get() + numChar * numNodes * numRates,
        rescaleBuffer.get() + numChar * numNodes * numRates
    );

    std::copy(currentConditionalLikelihoodFlags.get(),
        currentConditionalLikelihoodFlags.get() + numNodes,
        oldConditionalLikelihoodFlags.get()
    );
    
    oldPhylogeny = currentPhylogeny;
    oldPInvar = currentPInvar;
    oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;
    oldLnLikelihood = currentLnLikelihood;
    oldRates = currentRates;
    oldShape = currentShape;

    #if TIME_PROFILE == 1
    std::chrono::steady_clock::time_point postLikelihood = std::chrono::steady_clock::now();
    std::cout << "The forced likelihood update and accept compelted in " << std::chrono::duration_cast<std::chrono::milliseconds>(postTransProb - preTransProb).count() << "[milliseconds]" << std::endl;
    #endif
}

void Particle::writeToSerialized(SerializedParticle& sp){
    sp.newick = this->getNewick(); 
    sp.assignments = this->getAssignments();
    sp.stationaries = this->getCategories();
    sp.baseMatrix = this->getBaseMatrix(); 
    sp.pInvar = this->getPInvar(); 
    sp.shape = this->getShape();
}

std::vector<int> Particle::getAssignments() {
    std::vector<int> assignmentVector(numChar, -1);
    for(int i = 0; i < currentTransitionProbabilityClasses.size(); i++){
        auto cls = currentTransitionProbabilityClasses[i];
        for(int m : cls.members)
            assignmentVector[m] = i;
    }

    return assignmentVector;
}

std::vector<Eigen::Vector<double, 20>> Particle::getCategories() {
    std::vector<Eigen::Vector<double, 20>> categories;
    categories.reserve(currentTransitionProbabilityClasses.size());

    for(auto& c : currentTransitionProbabilityClasses)
        categories.push_back(c.stationaryDistribution);

    return categories;
}

void Particle::accept(){
    oldLnLikelihood = currentLnLikelihood;

    if(updateNNI || updateBranchLength || updateScaleSubtree || updateAdaptiveNNI){
        oldPhylogeny = currentPhylogeny;
    }

    std::copy(rescaleBuffer.get(), 
        rescaleBuffer.get() + numChar * numNodes * numRates,
        rescaleBuffer.get() + numChar * numNodes * numRates
    );

    std::copy(currentConditionalLikelihoodFlags.get(), 
        currentConditionalLikelihoodFlags.get() + numNodes,
        oldConditionalLikelihoodFlags.get()
    );
    
    if(updateBranchLength || updateStationary || updateScaleSubtree || updateRate || updateAdaptiveNNI){
        oldTransitionProbabilityClasses = currentTransitionProbabilityClasses;
        oldRates = currentRates;
        oldShape = currentShape;
    }

    oldPInvar = currentPInvar;

    updateStationary = false;
    updateNNI = false;
    updateAdaptiveNNI = false;
    updateBranchLength = false;
    updateScaleSubtree = false;
    updateInvar = false;
    updateRate = false;
}

void Particle::reject(){
    currentLnLikelihood = oldLnLikelihood;

    if(updateNNI || updateBranchLength || updateScaleSubtree || updateAdaptiveNNI){
        currentPhylogeny = oldPhylogeny;
    }

    std::copy(rescaleBuffer.get() + numChar * numNodes * numRates,
        rescaleBuffer.get() + 2 * numChar * numNodes * numRates,
        rescaleBuffer.get()
    );

    std::copy(oldConditionalLikelihoodFlags.get(), 
        oldConditionalLikelihoodFlags.get() + numNodes,
        currentConditionalLikelihoodFlags.get()
    );

    if(updateBranchLength || updateStationary || updateScaleSubtree || updateRate || updateAdaptiveNNI){
        currentTransitionProbabilityClasses = oldTransitionProbabilityClasses;
        currentRates = oldRates;
        currentShape = oldShape;
    }

    currentPInvar = oldPInvar;

    updateStationary = false;
    updateNNI = false;
    updateAdaptiveNNI = false;
    updateBranchLength = false;
    updateBranchLength = false;
    updateInvar = false;
    updateRate = false;
}

double Particle::lnLikelihood(){
    return currentLnLikelihood;
}

double Particle::lnPrior(){
    double lnP = 0.0;

    // Tree Prior
    double shape = static_cast<double>(numNodes - 1);
    double treeRate = 10.0;
    double lengthSum = 0.0;
    for(auto n : currentPhylogeny.getPostOrder()){
        lengthSum += n->branchLength;
    }
    lnP += boost::math::pdf(boost::math::gamma_distribution<double>(shape, 1.0/treeRate), lengthSum);

    // Shape prior
    lnP += boost::math::pdf(boost::math::gamma_distribution<double>(3.0, 0.5), currentShape);

    // For now we will assume flat priors on the stationary
    lnP += std::log(dppAlpha) * currentTransitionProbabilityClasses.size();
    for (auto& c : currentTransitionProbabilityClasses) {
        int memberCount = c.members.size();
        lnP += std::lgamma(memberCount);
    }
    lnP += std::lgamma(dppAlpha) - std::lgamma(dppAlpha + numChar);

    return lnP;
}

void Particle::setAssignments(std::vector<int>& assignments) {
    for (int i = 0; i < currentTransitionProbabilityClasses.size(); i++) {
        currentTransitionProbabilityClasses[i].members.clear();
        for (size_t m = 0; m < assignments.size(); m++){
            if (assignments[m] == i){
                currentTransitionProbabilityClasses[i].members.insert(m);
            }
        }
    }
}

void Particle::refreshLikelihood(bool forceUpdate){
    const auto postOrder = currentPhylogeny.getPostOrder();

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point preTPUpdate = std::chrono::steady_clock::now();
    #endif

    if(updateStationary || forceUpdate){ // Update all TPs of that class if that was the last move
        for(auto& c : currentTransitionProbabilityClasses){
            if(c.updated || forceUpdate){
                c.recomputeEigens();
                for(auto n : postOrder){
                    n->updateTP = false;
                    for(int r = 0; r < numRates; r++)
                        c.recomputeTransitionProbs(n->id, n->branchLength, r, currentRates[r]);
                }
                c.updated = false;
            }
        }
    }
    else if(updateBranchLength || updateScaleSubtree || updateRate){ // Update specific branch lengths if that was the last move
        for(auto n : postOrder){
            if(n->updateTP){
                n->updateTP = false;
                for(auto& c : currentTransitionProbabilityClasses){
                    for(int r = 0; r < numRates; r++){
                        c.recomputeTransitionProbs(n->id, n->branchLength, r, currentRates[r]);
                    }
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
        if(n->updateCL || forceUpdate){
            currentConditionalLikelihoodFlags[n->id] ^= 1;
        }
    }

    #if TIME_PROFILE==1
    std::chrono::steady_clock::time_point prePrune = std::chrono::steady_clock::now();
    #endif

    int nodeSpacer = numChar * numRates; // The length that spans a single node of the condL buffer
    int fullSpacer = numNodes * nodeSpacer; // The length that spans the whole condL buffer
    Eigen::Matrix<CL_TYPE, 20, 1> workBuffer;

    for(auto n : postOrder){
        int nIndex = n->id;
        uint8_t nWorkingSpace = currentConditionalLikelihoodFlags[nIndex];
        if((n->updateCL || forceUpdate) && !n->isTip){ // Tips never need their CL buffer updated because it just defined by the data
            auto pN = conditionaLikelihoodBuffer.get() + (nIndex * nodeSpacer) + (nWorkingSpace * fullSpacer);
            for(int i = 0; i < nodeSpacer; i++){
                pN[i].setOnes();
            }

            for(auto d : n->descendants){
                int dIndex = d->id;
                int pOffset = dIndex * numRates;
                uint8_t dWorkingSpace = currentConditionalLikelihoodFlags[dIndex];
                auto pD = conditionaLikelihoodBuffer.get() + (dIndex * nodeSpacer) + (dWorkingSpace * fullSpacer);

                /*
                    We are adopting a class-wise version of the inner loop. Indexing isn't as nice as incrementing a pointer
                    but when we have many classes we don't want to have to re-access a block of memory. Lets say that we have
                    100s of matrices - those aren't going to all fit in the cache. So if we alternate between matrix use due
                    to iterating over sites rather than classes, we risk additional cache misses
                */
                for(auto& tClass : currentTransitionProbabilityClasses){
                    for(int r = 0; r < numRates; r++){
                        Eigen::Matrix<CL_TYPE, 20, 20>& P = tClass.transitionProbabilities[pOffset + r];
                        for(int c : tClass.members){
                            auto& pDRef = pD[c * numRates + r];
                            auto& pNRef = pN[c * numRates + r];
                            workBuffer.noalias() = P * pDRef;
                            pNRef.array() *= workBuffer.array();
                        }
                    }
                }
            }
            double* rescalePointer = rescaleBuffer.get() + nodeSpacer * nIndex;
            std::fill(rescalePointer, rescalePointer + nodeSpacer, 0.0);

            for(int c = 0; c < nodeSpacer; c++){
                double max = static_cast<double>(pN->maxCoeff()); // We just show the cast here in case we are using single precision
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
    auto pR = conditionaLikelihoodBuffer.get() + rIndex * nodeSpacer + (dWorkingSpace * fullSpacer);
    
    double lnL = 0.0;
    std::vector<double> logs(numRates);

    double invInvarScaler = 1.0 - currentPInvar;

    for(auto& tClass : currentTransitionProbabilityClasses){
        #if MIXED_PRECISION
        auto stationaryVec = (tClass.stationaryDistribution).array().cast<CL_TYPE>(); // We cast here so we don't need to cast in the hot loop. We lose some precision but that is okay
        #else
        auto stationaryVec = (tClass.stationaryDistribution).array();
        #endif
        for(int c : tClass.members){
            // We need to use log-sum-exp to handle multiple rates
            double maxLogLike = -1.0 * INFINITY;

            for (int r = 0; r < numRates; r++) {
                double logLikeR = std::log((stationaryVec * (pR[c * numRates + r]).array()).sum());
                for (int n = 0; n < numNodes; n++)
                    logLikeR += rescaleBuffer[nodeSpacer * n + c * numRates + r];
                logs[r] = logLikeR;
                if (logLikeR > maxLogLike) maxLogLike = logLikeR;
            }

            double sumExp = 0.0;
            for (int r = 0; r < numRates; r++)
                sumExp += std::exp(logs[r] - maxLogLike);

            double logSiteLike = maxLogLike + std::log(sumExp);

            double finalSiteLike = std::exp(logSiteLike) * invInvarScaler
                + currentPInvar * stationaryVec[invariantCharacter[c]] * isInvariant[c];

            lnL += std::log(finalSiteLike);
        }
    }

    currentLnLikelihood = lnL;
}

double Particle::topologyMove(const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosterior){
    double hastings = 0.0;

    if(boost::random::uniform_01<double>{}(rng) < 0.5){
        hastings = currentPhylogeny.NNIMove(rng);
        updateNNI = true;
    }
    else{
        hastings = currentPhylogeny.adaptiveNNIMove(rng, aNNIEpsilon, splitPosterior);
        updateAdaptiveNNI = true;
    }

    return hastings;
}

double Particle::branchMove(){
    double hastings = 0.0;

    if(boost::random::uniform_01<double>{}(rng) < 0.75){
        hastings = currentPhylogeny.scaleBranchMove(rng, scaleDelta);
        updateBranchLength = true;
    }
    else {
        hastings = currentPhylogeny.scaleSubtreeMove(rng, subtreeScaleDelta);
        updateScaleSubtree = true;
    }

    return hastings;
}

double Particle::stationaryMove(){
    std::vector<double> weights;
    weights.reserve(currentTransitionProbabilityClasses.size());
    double total = 0.0;
    for(auto& c : currentTransitionProbabilityClasses){
        int classSize = c.members.size();
        total += classSize;
        weights.push_back(classSize);
    }

    double randomDraw = boost::random::uniform_01<double>{}(rng) * total;
    int randCategory = 0;
    double cumSum = 0.0;
    for(int i = 0; i < weights.size(); i++){
        cumSum += weights[i];
        if(cumSum >= randomDraw){
            randCategory = i;
            break;
        }
    }

    double hastings = currentTransitionProbabilityClasses[randCategory].dirichletSimplexMove(rng, stationaryAlpha);
    updateStationary = true;
    currentPhylogeny.updateAll();

    return hastings;
}

double Particle::invarMove(){
    double a = invarAlpha + 1.0;
    double b = (invarAlpha / currentPInvar) - a + 2.0;
    boost::random::beta_distribution<double> forwardDist{a, b};
    boost::math::beta_distribution<double> forwardDensity{a, b};
    double newPInvar = forwardDist(rng); 

    double a2 = invarAlpha + 1.0;
    double b2 = (invarAlpha / newPInvar) - a2 + 2.0;
    boost::math::beta_distribution<double> backwardDesnity{a2, b2};

    double forward = boost::math::pdf(forwardDensity, newPInvar);
    double backward = boost::math::pdf(backwardDesnity, currentPInvar);

    currentPInvar = newPInvar;

    updateInvar = true;

    return backward - forward;
}

double Particle::shapeMove(){
    double scale = std::exp(shapeDelta * (boost::random::uniform_01<double>{}(rng) - 0.5));
    double currentV = currentShape;
    double newV = currentV * scale;
    currentShape = newV;

    updateRate = true;
    currentPhylogeny.updateAll();
    discretizeGamma(currentRates, currentShape, numRates);

    return std::log(scale);
}

double Particle::gibbsPartitionMove(double tempering){
    boost::random::uniform_01<double> unif{};

    const auto postOrder = currentPhylogeny.getPostOrder();

    int numAux = 5;
    int numIter = static_cast<int>(numChar/2.0);

    double alphaSplit = std::log(dppAlpha/numAux);

    int bufferSize = (2 * numAux + currentTransitionProbabilityClasses.size());
    auto tempCLBuffer = new Eigen::Vector<CL_TYPE, 20>[numNodes * numRates * bufferSize];
    auto tempRescaleBuffer = new double[numNodes * numRates * bufferSize];
    for(int i = 0; i < numNodes * numRates * bufferSize; i++){
        tempRescaleBuffer[i] = 0.0;
        tempCLBuffer[i] = Eigen::Vector<CL_TYPE, 20>::Zero();
    }

    // Choosing to update randomly seems like it is the best strategy?
    for(int iter = 0; iter < numIter; iter++){
        #if TIME_PROFILE==1
        std::chrono::steady_clock::time_point preIter = std::chrono::steady_clock::now();
        #endif
        int randomSite = static_cast<int>(unif(rng) * numChar);
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
            auto auxCat = TransitionProbabilityClass(rng, numNodes, numRates, baseMatrix.get());
            auxCat.recomputeEigens();
            for(auto node : postOrder){
                node->updateTP = false;
                for(int r = 0; r < numRates; r++){
                    auxCat.recomputeTransitionProbs(node->id, node->branchLength, r, currentRates[r]);
                }
            }
            currentTransitionProbabilityClasses.push_back(auxCat);
        }

        int catSize = currentTransitionProbabilityClasses.size();

        Eigen::Matrix<CL_TYPE,20,1> workBuffer;

        // We try to give ourselves a big enough buffer at the beginning, but if it gets too small, update it.
        if(bufferSize < catSize){
            delete [] tempCLBuffer;
            delete [] tempRescaleBuffer;

            bufferSize = (2 * numAux + catSize);
            tempCLBuffer = new Eigen::Vector<CL_TYPE, 20>[numNodes * numRates * bufferSize];
            tempRescaleBuffer = new double[numNodes * numRates * bufferSize];
            for(int i = 0; i < numNodes * numRates * bufferSize; i++){
                tempRescaleBuffer[i] = 0.0;
                tempCLBuffer[i] = Eigen::Vector<CL_TYPE, 20>::Zero();
            }
        }

        int nodeSpacer = bufferSize * numRates;
        int activeNodeSpacer = catSize * numRates;
        int fullSpacer = numNodes * nodeSpacer;

        // Run the pruning algorithm
        for(auto n : postOrder){
            int nIndex = n->id;
            auto pN = tempCLBuffer + nIndex * nodeSpacer;
            if(!n->isTip){
                for(int i = 0; i < activeNodeSpacer; i++){
                    pN[i].setOnes();
                }

                for(auto d : n->descendants){
                    int dIndex = d->id;
                    int pOffset = dIndex * numRates;
                    auto pD = tempCLBuffer+ dIndex * nodeSpacer;
                    
                    for(int c = 0; c < catSize; c++){
                        for(int r = 0; r < numRates; r++){
                            Eigen::Matrix<CL_TYPE, 20, 20>& P = currentTransitionProbabilityClasses[c].transitionProbabilities[pOffset + r];
                            auto& pDRef = pD[c * numRates + r];
                            auto& pNRef = pN[c * numRates + r];
                            workBuffer.noalias() = P * pDRef;
                            pNRef.array() *= workBuffer.array();
                        }
                    }
                }
                double* rescalePointer = tempRescaleBuffer + activeNodeSpacer * nIndex;
                std::fill(rescalePointer, rescalePointer + activeNodeSpacer, 0.0);

                for(int c = 0; c < activeNodeSpacer; c++){
                    double max = static_cast<double>(pN->maxCoeff());
                    *pN /= max;
                    *rescalePointer = std::log(max);

                    rescalePointer++;
                    pN++;
                }
            }
            else{
                for(int c = 0; c < catSize; c++){
                    for(int r = 0; r < numRates; r++){
                        pN[c*numRates + r].noalias() = *(conditionaLikelihoodBuffer.get() + nIndex * numChar * numRates + randomSite);
                    }
                }
            }
        }

        int rIndex = currentPhylogeny.getRoot()->id;
        auto pR = tempCLBuffer + rIndex * nodeSpacer;
        
        std::vector<double> likelihoodVec;
        likelihoodVec.reserve(catSize);
        std::vector<double> logs(numRates);
        double invInvarScaler = 1.0 - currentPInvar;

        for(int c = 0; c < catSize; c++){
            #if MIXED_PRECISION
            auto stationaryVec = (currentTransitionProbabilityClasses[c].stationaryDistribution).array().cast<CL_TYPE>();
            #else
            auto stationaryVec = (currentTransitionProbabilityClasses[c].stationaryDistribution).array();
            #endif

            double maxLogLike = -std::numeric_limits<double>::infinity();

            for (int r = 0; r < numRates; r++) {
                double logLikeR = std::log((stationaryVec * (pR[c*numRates + r]).array()).sum());
                for (int n = 0; n < numNodes; n++)
                    logLikeR += tempRescaleBuffer[nodeSpacer * n + c * numRates + r];
                logs[r] = logLikeR;
                if (logLikeR > maxLogLike) maxLogLike = logLikeR;
            }

            double sumExp = 0.0;
            for (int r = 0; r < numRates; r++)
                sumExp += std::exp(logs[r] - maxLogLike);

            double logSiteLike = maxLogLike + std::log(sumExp);
            double finalSiteLike = std::exp(logSiteLike) * invInvarScaler
                + currentPInvar * stationaryVec[invariantCharacter[randomSite]] * isInvariant[randomSite];
            finalSiteLike = std::log(finalSiteLike) * tempering;

            if(c < catSize - numAux){
                likelihoodVec.push_back(finalSiteLike + std::log(currentTransitionProbabilityClasses[c].members.size()));
            }
            else {
                likelihoodVec.push_back(finalSiteLike + alphaSplit);
            }
        }

        double maxL = *std::max_element(likelihoodVec.begin(), likelihoodVec.end());
        double total = 0.0;
        for(double& d : likelihoodVec){
            d -= maxL;
            d = std::exp(d);
            total += d;
        }

        double categoryDraw = total * unif(rng);

        auto start = currentTransitionProbabilityClasses.begin() + (catSize - numAux);

        total = 0.0;
        for(int i = 0; i < likelihoodVec.size(); i++){
            total += likelihoodVec[i];
            if(total > categoryDraw){
                if(i < catSize - numAux) { //It already exists
                    currentTransitionProbabilityClasses.erase(start, currentTransitionProbabilityClasses.end());
                    currentTransitionProbabilityClasses[i].members.insert(randomSite);
                }
                else {
                    // Delete the num aux other than the one we want.
                    auto skip  = currentTransitionProbabilityClasses.begin() + i;
                    if(skip >= start){
                        currentTransitionProbabilityClasses.erase(start, skip);
                        skip = currentTransitionProbabilityClasses.begin() + (catSize - numAux); // New position of the skipped element
                        currentTransitionProbabilityClasses.erase(skip + 1, currentTransitionProbabilityClasses.end());
                    }
                    else{
                        currentTransitionProbabilityClasses.erase(start, currentTransitionProbabilityClasses.end());
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