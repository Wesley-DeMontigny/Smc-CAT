#ifndef TRANSITION_PROBABILITY_CLASS_HPP
#define TRANSITION_PROBABILITY_CLASS_HPP
#include <eigen3/Eigen/Dense>
#include <memory>
#include <random>
#include <set>
#include "core/Types.hpp"

struct TransitionProbabilityClass {
    TransitionProbabilityClass(int n, int c, Eigen::Matrix<double, 20, 20>* bM);

    std::vector<Eigen::Matrix<CL_TYPE, 20, 20>> transitionProbabilities; // Indexed by node
    
    Eigen::Matrix<double, 20, 20>* baseMatrix;
    Eigen::Vector<double, 20> stationaryDistribution;

    std::set<int> members;

    void recomputeEigens();
    void recomputeTransitionProbs(int n, double t, int c, double r);
    double dirichletSimplexMove(double alpha);

    static Eigen::Vector<double, 20> sampleStationary(const Eigen::Vector<double, 20>& alpha);
    static double stationarylnPdf(const Eigen::Vector<double, 20>& concentration, const Eigen::Vector<double, 20>& x);

    bool updated;
    private:
        Eigen::DiagonalMatrix<Eigen::dcomplex, 20> workingDiag;
        Eigen::Matrix<Eigen::dcomplex, 20, 20> workingMatrix1;
        Eigen::Matrix<Eigen::dcomplex, 20, 20> workingMatrix2;

        int numRates;
        Eigen::Matrix<std::complex<double>, 20, 20> eigenVectors;
        Eigen::Matrix<std::complex<double>, 20, 20> inverseEigenVectors;
        Eigen::Vector<std::complex<double>, 20> eigenValues;
};

#endif