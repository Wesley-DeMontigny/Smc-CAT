#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include "core/Miscellaneous.hpp"
#include "TransitionProbabilityClass.hpp"
#include "Tree.hpp"
#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <eigen3/Eigen/Dense>
#include <memory>

class Alignment;
struct SerializedParticle;

/**
 * @brief 
 * 
 */
enum UpdateType{
    ADAPTIVE_NNI = 0,
    NNI = 1,
    SPR = 2,
    BRANCH_LENGTH = 3,
    SCALE_SUBTREE = 4,
    INVAR = 5,
    RATE_SHAPE = 6,
    STATIONARY_SINGLE = 7,
    RATE_MATRIX_SINGLE = 8,
    RATE_MATRIX_FULL = 9,
    NO_MOVE = 10
};

static const std::array<std::string, 10> MoveNames = {
    "Adaptive NNI", 
    "NNI",
    "SPR",
    "Branch Length",
    "Scale Subtree",
    "Proportion Invariant",
    "Rate Shape",
    "Stationary",
    "Rate Matrix Single",
    "Rate Matrix Full"
};

/**
 * @brief 
 */
class Particle {
    public:
        Particle(void)=delete;
        Particle(int seed, Alignment& aln, int nR=1, bool initInvar=false, bool lg=true, bool cat=true);
        
        double getPInvar() const { return currentPInvar; }
        double getShape() const { return currentShape; }
        double gibbsPartitionMove(double tempering); // Returns infinity in case it is in an MH setting
        double invarMove();
        double lnLikelihood();
        double lnPrior();
        double rateMatrixMove();
        double shapeMove();
        double stationaryMove();
        double treeMove(const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosterior);
        boost::random::mt19937& getRng(){ return rng; }
        Eigen::Vector<double, 190> getBaseMatrix() const { return *currentBaseMatrix; }
        int getNumCategories() { return currentTransitionProbabilityClasses.size(); }
        int getNumNodes() const { return numNodes; }
        int getNumRates() const { return numRates; }
        std::set<boost::dynamic_bitset<>> getSplitSet() { return currentPhylogeny.getSplitSet(); }
        std::unordered_map<boost::dynamic_bitset<>, double> getSplitBranchMap() { return currentPhylogeny.getSplitBranchMap(); }
        std::string getNewick() const { return currentPhylogeny.generateNewick(); }
        std::string getNewick(const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities) const { return currentPhylogeny.generateNewick(splitPosteriorProbabilities); }
        std::vector<double> getRates() const { return currentRates; }
        std::vector<Eigen::Vector<double, 20>> getCategories() const;
        std::vector<int> getAssignments() const;
        void accept();
        void copy(Particle& p);
        void copyFromSerialized(SerializedParticle& sp);
        void initialize(bool initInvar=false, bool initCat=true);
        void refreshLikelihood(bool forceUpdate = false); // Refreshes the likelihood and stores it in the currentLnLikelihood variable
        void reject();
        void setAssignments(std::vector<int>& assignments);
        void setInvariance(double i) { currentPInvar = i; }
        void writeToSerialized(SerializedParticle& sp);

        std::array<int, 10> moveCount;
        std::array<int, 10> acceptCount;
        double aNNIEpsilon = 0.01; // The offset for probabilities in the adaptive NNI. This can be thought of as the base tendency to select an edge with posterior of 1.0
        double shapeDelta = 1.0; // Delta to scale the shape of the gamma distribution
        double scaleDelta = 1.0; // Delta to scale an individual branch length
        double subtreeScaleDelta = 1.0; // Delta to scale whole subtrees
        double stationaryDelta = 0.25; // Delta parameter for proposals on the stationary
        double rateMatrixDelta = 0.25; // Delta parameter for proposals on the rate matrix
        Eigen::MatrixXd rateMatrixCholesky = Eigen::MatrixXd::Identity(190, 190) * 0.5; // The Cholesky factor for entries of the rate matrix
        double invarAlpha = 100.0; // Concentration parameter for the beta simplex proposals on invar
    private:
        Alignment& aln;
        bool usingLG;
        boost::random::mt19937 rng;
        const int numChar;
        const int numNodes;
        const int numRates;
        const int numTaxa;
        double currentLnLikelihood = 0.0;
        double currentPInvar = 0.0;
        double currentShape = 1.0;
        double dppAlpha = 0.01;
        double oldLnLikelihood = 0.0;
        double oldPInvar = 0.0;
        double oldShape = 1.0;
        std::unique_ptr<double[]> isInvariant;
        std::unique_ptr<double[]> rescaleBuffer; // Contains all the rescale values we computed. Should be size NumNodes x NumSites x 2.
        std::unique_ptr<Eigen::Vector<double, 190>> currentBaseMatrix;
        std::unique_ptr<Eigen::Vector<double, 190>> oldBaseMatrix;
        std::unique_ptr<Eigen::Vector<CL_TYPE, 20>[]> conditionaLikelihoodBuffer; // Contains all the conditional likelihoods for each node (rescaled). Should be size NumSites x NumNodes x 2
        std::unique_ptr<int[]> invariantCharacter;
        std::unique_ptr<uint8_t[]> currentConditionalLikelihoodFlags; // To stop us from having to swap the whole memory space, we just keep a working space flag for each node
        std::unique_ptr<uint8_t[]> oldConditionalLikelihoodFlags; // Swap back flags if rejected
        std::vector<double> currentRates;
        std::vector<double> oldRates;
        std::vector<TransitionProbabilityClass> currentTransitionProbabilityClasses; // Contains all of the current DPP categories
        std::vector<TransitionProbabilityClass> oldTransitionProbabilityClasses; // Memory of the DPP categories to restore
        Tree currentPhylogeny; // We need the phylogenies to be evaluated before the numNodes
        Tree oldPhylogeny;
        UpdateType currentMove = UpdateType::NO_MOVE;
};

#endif