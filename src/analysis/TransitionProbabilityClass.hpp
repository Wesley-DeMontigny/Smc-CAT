#ifndef TRANSITION_PROBABILITY_CLASS_HPP
#define TRANSITION_PROBABILITY_CLASS_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include <random>
#include <set>

struct TransitionProbabilityClass {
    TransitionProbabilityClass(int n, int c, Eigen::Matrix<double, 20, 20>* bM);

    std::vector<Eigen::Matrix<double, 20, 20>> transitionProbabilities; // Indexed by node
    Eigen::Matrix<std::complex<double>, 20, 20> eigenVectors;
    Eigen::Matrix<std::complex<double>, 20, 20> inverseEigenVectors;
    Eigen::Vector<std::complex<double>, 20> eigenValues;
    
    Eigen::Matrix<double, 20, 20>* baseMatrix;
    Eigen::Vector<double, 20> stationaryDistribution;

    std::set<int> members;

    void recomputeEigens();
    void recomputeTransitionProbs(int n, double t, int c, double r);
    double dirichletSimplexMove(double alpha, std::mt19937& gen);

    static Eigen::Vector<double, 20> sampleStationary(Eigen::Vector<double, 20> alpha, std::mt19937& gen);

    bool updated;
    int numRates;
};

#endif