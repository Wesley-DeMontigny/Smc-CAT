#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include "core/Types.hpp"
#include "TransitionProbabilityClass.hpp"
#include "Tree.hpp"

class Alignment;
class SerializedParticle;

/*

*/
class Particle {
    public:
        Particle(void)=delete;
        Particle(Alignment& aln, int nR=1, bool initInvar=false);
        
        double branchMove();
        double getPInvar() { return currentPInvar; }
        double getShape() { return currentShape; }
        double gibbsPartitionMove(double tempering); // Returns infinity in case it is in an MH setting
        double invarMove();
        double lnLikelihood();
        double lnPrior();
        double shapeMove();
        double stationaryMove();
        double topologyMove(const std::unordered_map<std::string, double>& splitPosterior);
        Eigen::Matrix<double, 20, 20> getBaseMatrix() { return *baseMatrix; }
        int getNumCategories() { return currentTransitionProbabilityClasses.size(); }
        int getNumNodes() { return numNodes; }
        int getNumRates() { return numRates; }
        std::set<std::string> getSplits() { return currentPhylogeny.getSplits(); }
        std::string getNewick() { return currentPhylogeny.generateNewick(); }
        std::string getNewick(std::unordered_map<std::string, double>& splitPosteriorProbabilities) { return currentPhylogeny.generateNewick(splitPosteriorProbabilities); }
        std::vector<double> getRates() { return currentRates; }
        std::vector<Eigen::Vector<double, 20>> getCategories();
        std::vector<int> getAssignments();
        void accept();
        void copy(Particle& p);
        void copyFromSerialized(SerializedParticle& sp);
        void initialize(bool initInvar=false);
        void read(int id, std::string& dir);
        void refreshLikelihood(bool forceUpdate = false); // Refreshes the likelihood and stores it in the currentLnLikelihood variable
        void reject();
        void setAssignments(std::vector<int>& assignments);
        void setInvariance(double i) { currentPInvar = i; }
        void tune(); // Tune the proposal parameters
        void write(int id, std::string& dir);
        void writeToSerialized(SerializedParticle& sp);

        // Variables to keep track of acceptance rates so we can track performance
        int proposedStationary = 0;
        int acceptedStationary = 0;
        int proposedNNI = 0;
        int acceptedNNI = 0;
        int proposedAdaptiveNNI = 0;
        int acceptedAdaptiveNNI = 0;
        int proposedBranchLength = 0;
        int acceptedBranchLength = 0;
        int proposedSubtreeScale = 0;
        int acceptedSubtreeScale = 0;
        int proposedInvar = 0;
        int acceptedInvar = 0;
        int proposedRate = 0;
        int acceptedRate = 0;

        double aNNIEpsilon = 0.001; // The offset for probabilities in the adaptive NNI. This can be thought of as the probability of selecting an edge with posterior of 1.0
        double shapeDelta = 1.0; // Delta to scale the shape of the gamma distribution
        double scaleDelta = 1.0; // Delta to scale an individual branch length
        double subtreeScaleDelta = 1.0; // Delta to scale whole subtrees
        double stationaryAlpha = 10000.0; // Concentration parameter for dirichlet simplex proposals
        double invarAlpha = 100.0; // Concentration parameter for the beta simplex proposals on invar
    private:
        Tree currentPhylogeny; // We need the phylogenies to be evaluated before the numNodes
        Tree oldPhylogeny;
        std::unique_ptr<Eigen::Matrix<double, 20, 20>> baseMatrix;
        Alignment& aln;

        int numChar;
        int numTaxa;
        int numNodes;

        double currentLnLikelihood = 0.0;
        double oldLnLikelihood = 0.0;

        bool updateStationary = false;
        bool updateNNI = false;
        bool updateAdaptiveNNI = false;
        bool updateBranchLength = false;
        bool updateScaleSubtree = false;
        bool updateInvar = false;
        bool updateRate = false;

        std::unique_ptr<uint8_t[]> currentConditionalLikelihoodFlags; // To stop us from having to swap the whole memory space, we just keep a working space flag for each node
        std::unique_ptr<uint8_t[]> oldConditionalLikelihoodFlags; // Swap back flags if rejected

        std::unique_ptr<Eigen::Vector<CL_TYPE, 20>[]> conditionaLikelihoodBuffer; // Contains all the conditional likelihoods for each node (rescaled). Should be size NumSites x NumNodes x 2
        std::unique_ptr<double[]> rescaleBuffer; // Contains all the rescale values we computed. Should be size NumNodes x NumSites x 2.
        
        std::unique_ptr<double[]> isInvariant;
        std::unique_ptr<int[]> invariantCharacter;
        double currentPInvar = 0.0;
        double oldPInvar = 0.0;

        int numRates;
        double currentShape = 1.0;
        double oldShape = 1.0;
        std::vector<double> currentRates;
        std::vector<double> oldRates;

        std::vector<TransitionProbabilityClass> currentTransitionProbabilityClasses; // Contains all of the current DPP categories
        std::vector<TransitionProbabilityClass> oldTransitionProbabilityClasses; // Memory of the DPP categories to restore
        double dppAlpha = 0.01;
};

#endif