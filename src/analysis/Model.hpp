#ifndef LIKELIHOOD_HPP
#define LIKELIHOOD_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include "TransitionProbabilityClass.hpp"
#include "Tree.hpp"

class Alignment;

/*

*/
class Model {
    public:
        Model(void)=delete;
        Model(Alignment& aln);

        void refreshLikelihood();

        double lnPrior();
        double lnLikelihood();

        void accept();
        void reject();
        
        double topologyMove();
        double branchMove();
        double stationaryMove();

        void tune(); // Tune the proposal parameters

        std::string getNewick() {return currentPhylogeny.generateNewick();}

        // Variables to keep track of acceptance rates so we can tune and track performance
        int proposedStationary = 0;
        int acceptedStationary = 0;
        int proposedNNI = 0;
        int acceptedNNI = 0;
        int proposedBranchLength = 0;
        int acceptedBranchLength = 0;
        int proposedSubtreeScale = 0;
        int acceptedSubtreeScale = 0;
        int proposedAssignments = 0;
        int acceptedAssignments = 0;

        double scaleDelta = 1.0; // Delta to scale an individual branch length
        double subtreeScaleDelta = 1.0; // Delta to scale whole subtrees
        double stationaryAlpha = 500.0; // Concentration parameter for dirichlet simplex proposals
    private:
        Tree currentPhylogeny; // We need the phylogenies to be evaluated before the numNodes
        Tree oldPhylogeny;
        std::unique_ptr<Eigen::Matrix<double, 20, 20>> baseMatrix;

        int numChar;
        int numTaxa;
        int numNodes;

        double currentLnLikelihood = 0.0;
        double oldLnLikelihood = 0.0;

        bool updateStationary = false;
        bool updateNNI = false;
        bool updateBranchLength = false;
        bool updateScaleSubtree = false;
        bool updateAssignments = false;

        std::unique_ptr<uint8_t[]> currentConditionalLikelihoodFlags; // To stop us from having to swap the whole memory space, we just keep a working space flag for each node
        std::unique_ptr<uint8_t[]> oldConditionalLikelihoodFlags; // Swap back flags if rejected
        std::unique_ptr<int[]> currentClassAssignments; // Working space for the class assignments
        std::unique_ptr<int[]> oldClassAssignments; // Memory for the class assignments

        std::unique_ptr<Eigen::Vector<double, 20>[]> conditionaLikelihoodBuffer; // Contains all the conditional likelihoods for each node (rescaled). Should be size NumSites x NumNodes x 2
        std::unique_ptr<double[]> rescaleBuffer; // Contains all the rescale values we computed. Should be size NumNodes x NumSites x 2.
        std::vector<TransitionProbabilityClass> currentTransitionProbabilityClasses; // Contains all of the current DPP categories
        std::vector<TransitionProbabilityClass> oldTransitionProbabilityClasses; // Memory of the DPP categories to restore
};

#endif