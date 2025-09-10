#ifndef LIKELIHOOD_HPP
#define LIKELIHOOD_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include "TransitionProbabilityClass.hpp"

class Alignment;
class Tree;


/*

*/
class Model {
    public:
        Model(void)=delete;
        Model(Tree& t, Alignment& a, Eigen::Matrix<double, 20, 20> bM);

        void refreshLikelihood();

        double lnPrior();
        double lnLikelihood();

        void accept();
        void reject();
        
        void updateAssignments(int numUpdates);
        double updateStationary();

        void tune(); // Tune the proposal parameters

        int proposedStationary = 0;
        int acceptedStationary = 0;

        double stationaryAlpha = 100.0; // Concentration parameter for dirichlet simplex proposals
        double stationaryOffset = 0.01; // Offset parameter for dirichlet simplex proposals
    private:
        Tree& phylogeny;
        Eigen::Matrix<double, 20, 20> baseMatrix;
        int numChar;
        int numTaxa;
        int numNodes;

        double currentLnLikelihood = 0.0;
        double oldLnLikelihood = 0.0;

        int proposalID = -1;

        std::unique_ptr<uint8_t[]> currentConditionalLikelihoodFlags; // To stop us from having to swap the whole memory space, we just keep a working space flag for each node
        std::unique_ptr<uint8_t[]> oldConditionalLikelihoodFlags; // Swap back flags if rejected
        std::unique_ptr<int[]> currentClassAssignments; // Working space for the class assignments
        std::unique_ptr<int[]> oldClassAssignments; // Memory for the class assignments

        std::unique_ptr<Eigen::Vector<double, 20>[]> conditionaLikelihoodBuffer; // Contains all the conditional likelihoods for each node (rescaled). Should be size NumSites x NumNodes x 2
        std::unique_ptr<double[]> rescaleBuffer; // Contains all the rescale values we computed. Should be size NumNodes x NumSites x 2. Buffer selection should be by the CL Flag
        std::vector<TransitionProbabilityClass> currentTransitionProbabilityClasses; // Contains all of the current DPP categories
        std::vector<TransitionProbabilityClass> oldTransitionProbabilityClasses; // Memory of the DPP categories to restore
};

#endif