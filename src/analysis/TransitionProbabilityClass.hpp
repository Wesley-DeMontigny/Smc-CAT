#ifndef TRANSITION_PROBABILITY_CLASS_HPP
#define TRANSITION_PROBABILITY_CLASS_HPP
#include <eigen3/Eigen/Dense>

struct TransitionProbabilityClass {

    std::vector<Eigen::Matrix<double, 20, 20>> transitionProbabilities;
    Eigen::Matrix<std::complex<double>, 20, 20> eigenVectors;
    Eigen::Matrix<std::complex<double>, 20, 20> inverseEigenVectors;
    Eigen::Vector<std::complex<double>, 20> eigenValues;
    
    Eigen::Matrix<double, 20, 20>& baseMatrix;
    Eigen::Vector<double, 20> stationaryDistribution;

    void recomputeEigens();
    void recomputeTransitionProbs(int n, double t, double r);

    bool updated;

    TransitionProbabilityClass(int n, Eigen::Matrix<double, 20, 20>& bM);
};

#endif