#ifndef TRANSITION_PROBABILITY_CLASS_HPP
#define TRANSITION_PROBABILITY_CLASS_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include <random>

struct TransitionProbabilityClass {
    TransitionProbabilityClass(int n, std::shared_ptr<Eigen::Matrix<double, 20, 20>> bM);

    std::vector<Eigen::Matrix<double, 20, 20>> transitionProbabilities; // Indexed by node
    Eigen::Matrix<std::complex<double>, 20, 20> eigenVectors;
    Eigen::Matrix<std::complex<double>, 20, 20> inverseEigenVectors;
    Eigen::Vector<std::complex<double>, 20> eigenValues;
    
    std::shared_ptr<Eigen::Matrix<double, 20, 20>> baseMatrix;
    Eigen::Vector<double, 20> stationaryDistribution;

    void recomputeEigens();
    void recomputeTransitionProbs(int n, double t, double r);
    double dirichletSimplexMove(double alpha, double offset, std::mt19937& gen);

    bool updated;
};

#endif