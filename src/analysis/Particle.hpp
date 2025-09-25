#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include "TransitionProbabilityClass.hpp"
#include "Tree.hpp"

class Alignment;

/*

*/
class Particle {
    public:
        Particle(void)=delete;
        Particle(Alignment& aln);

        void initialize();

        void refreshLikelihood();

        double lnPrior();
        double lnLikelihood();

        void accept();
        void reject();
        
        double topologyMove();
        double branchMove();
        double stationaryMove();
        double invarMove();
        double gibbsPartitionMove(double tempering);

        void tune(); // Tune the proposal parameters

        std::string getNewick() { return currentPhylogeny.generateNewick(); }
        int getNumCategories() { return currentTransitionProbabilityClasses.size(); }
        int getNumNodes() { return numNodes; }

        void write(const int id, const std::string& dir);
        void read(const int id, const std::string& dir);

        // Variables to keep track of acceptance rates so we can tune and track performance
        int proposedStationary = 0;
        int acceptedStationary = 0;
        int proposedNNI = 0;
        int acceptedNNI = 0;
        int proposedBranchLength = 0;
        int acceptedBranchLength = 0;
        int proposedSubtreeScale = 0;
        int acceptedSubtreeScale = 0;
        int proposedInvar = 0;
        int acceptedInvar = 0;

        double scaleDelta = 1.0; // Delta to scale an individual branch length
        double subtreeScaleDelta = 1.0; // Delta to scale whole subtrees
        double stationaryAlpha = 500.0; // Concentration parameter for dirichlet simplex proposals
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
        bool updateBranchLength = false;
        bool updateScaleSubtree = false;
        bool updateInvar = false;

        std::unique_ptr<uint8_t[]> currentConditionalLikelihoodFlags; // To stop us from having to swap the whole memory space, we just keep a working space flag for each node
        std::unique_ptr<uint8_t[]> oldConditionalLikelihoodFlags; // Swap back flags if rejected

        std::unique_ptr<Eigen::Vector<double, 20>[]> conditionaLikelihoodBuffer; // Contains all the conditional likelihoods for each node (rescaled). Should be size NumSites x NumNodes x 2
        std::unique_ptr<double[]> rescaleBuffer; // Contains all the rescale values we computed. Should be size NumNodes x NumSites x 2.
        
        std::unique_ptr<double[]> isInvariant;
        std::unique_ptr<int[]> invariantCharacter;
        double currentPInvar = 0.1;
        double oldPInvar = 0.1;

        std::vector<TransitionProbabilityClass> currentTransitionProbabilityClasses; // Contains all of the current DPP categories
        std::vector<TransitionProbabilityClass> oldTransitionProbabilityClasses; // Memory of the DPP categories to restore
        double dppAlpha = 0.5;
};

#endif