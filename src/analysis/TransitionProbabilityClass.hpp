#ifndef TRANSITION_PROBABILITY_CLASS_HPP
#define TRANSITION_PROBABILITY_CLASS_HPP
#include "core/Miscellaneous.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <set>

/**
 * @brief 
 * 
 */
struct TransitionProbabilityClass {
    public:
        TransitionProbabilityClass(void)=delete;
        TransitionProbabilityClass(boost::random::mt19937& rng, int n, int c, Eigen::Vector<double, 190>* bM);

        void recomputeEigens();
        void recomputeTransitionProbs(int n, double t, int c, double r);
        void normalizeStationary();
        double stationaryMove(boost::random::mt19937& rng, double delta);
        double lnPrior();
        static Eigen::Vector<double, 20> sampleStationaryLogits(boost::random::mt19937& rng);
        static double stationarylnPdf(const Eigen::Vector<double, 20>& x);

        std::vector<Eigen::Matrix<CL_TYPE, 20, 20>> transitionProbabilities; // Indexed by node
        Eigen::Vector<double, 190>* baseMatrix;
        Eigen::Vector<double, 20> stationaryLogits;
        Eigen::Vector<double, 20> stationaryDistribution;
        Eigen::Matrix<std::complex<double>, 20, 20> eigenVectors;
        Eigen::Matrix<std::complex<double>, 20, 20> inverseEigenVectors;
        Eigen::Vector<std::complex<double>, 20> eigenValues;
        std::set<int> members;
        bool updated;
   private:
        Eigen::DiagonalMatrix<Eigen::dcomplex, 20> workingDiag;
        Eigen::Matrix<Eigen::dcomplex, 20, 20> workingMatrix1;
        Eigen::Matrix<Eigen::dcomplex, 20, 20> workingMatrix2;
        int numRates;
};

#endif