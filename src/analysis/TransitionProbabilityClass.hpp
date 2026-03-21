#ifndef TRANSITION_PROBABILITY_CLASS_HPP
#define TRANSITION_PROBABILITY_CLASS_HPP
#include "misc/Miscellaneous.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <set>

/**
 * @brief 
 * 
 */
class TransitionProbabilityClass {
    public:
        TransitionProbabilityClass(void)=delete;
        TransitionProbabilityClass(boost::random::mt19937& rng, int n, int c, Eigen::Vector<double, 190>* bM);

        void recomputeEigens();
        void recomputeTransitionProbs(int n, double t, int c, double r);
        void normalizeStationary();
        double stationaryMove(boost::random::mt19937& rng, double delta);
        double lnPrior();
        static Eigen::Vector<double, 20> sampleStationaryLogits(boost::random::mt19937& rng);

        std::vector<Eigen::Matrix<CL_TYPE, 20, 20>> transitionProbabilities; // Indexed by node
        Eigen::Vector<double, 190>* baseMatrix;
        Eigen::Vector<double, 20> stationaryLogits;
        Eigen::Vector<double, 20> stationaryDistribution;
        Eigen::Matrix<double, 20, 20> symEigenVectors;
        Eigen::Vector<double, 20> symEigenValues;
        std::set<int> members;
        bool updated;
   private:
        Eigen::Vector<double, 20> sqrtPi;
        Eigen::Vector<double, 20> invSqrtPi;
        int numRates;
};

#endif
