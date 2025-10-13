#include "TransitionProbabilityClass.hpp"
#include <core/Math.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>

TransitionProbabilityClass::TransitionProbabilityClass(int n, int c, Eigen::Matrix<double, 20, 20>* bM) : baseMatrix(bM), updated(false), numRates(c) {
    // Initialize a buffer of transition probabilities for each branch
    transitionProbabilities.reserve(n * numRates);
    for(int i = 0; i < n * numRates; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<double, 20, 20>::Zero() 
        );
    }

    auto generator = std::mt19937(std::random_device{}());
    stationaryDistribution = sampleStationary(Eigen::Vector<double, 20>::Ones(), generator);
}

Eigen::Vector<double, 20> TransitionProbabilityClass::sampleStationary(Eigen::Vector<double, 20> alpha, std::mt19937& gen){
    Eigen::Vector<double, 20> stationaryDistribution = Eigen::Vector<double, 20>::Zero();

    double denom = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        std::gamma_distribution<double> gammaDist(alpha(i), 1.0);
        double randGamma = gammaDist(gen);
        stationaryDistribution(i) = randGamma;
        denom += randGamma;
    }

    return stationaryDistribution / denom;
}


void TransitionProbabilityClass::recomputeEigens(){
Eigen::Matrix<double,20,20> Q = (*baseMatrix) * stationaryDistribution.asDiagonal();

    for (int i = 0; i < 20; i++) {
        double offDiag = 0.0;
        for (int j = 0; j < 20; j++) {
            if (j != i) offDiag += Q(i,j);
        }
        Q(i,i) = -offDiag;
    }

    // rescale mean rate to 1.0
    double meanRate = 0.0;
    for (int i = 0; i < 20; i++) {
        meanRate += -Q(i,i) * stationaryDistribution(i);
    }
    Q /= meanRate;

    Eigen::EigenSolver<Eigen::Matrix<double, 20, 20>> eigenSolver(Q);
    eigenValues = eigenSolver.eigenvalues();
    eigenVectors = eigenSolver.eigenvectors();
    inverseEigenVectors = eigenVectors.inverse();
}

void TransitionProbabilityClass::recomputeTransitionProbs(int n, double t, int c, double r){
    Eigen::DiagonalMatrix<std::complex<double>,20> diag;
    for(int i = 0; i < 20; i++)
        diag.diagonal()(i) = std::exp(eigenValues(i) * t * r);

    Eigen::Matrix<double, 20, 20> transProb = (eigenVectors * diag * inverseEigenVectors).real();

    transitionProbabilities[n*numRates + c] = transProb;
}

double TransitionProbabilityClass::dirichletSimplexMove(double alpha, std::mt19937& gen){
    Eigen::Vector<double, 20> concentrations = stationaryDistribution;
    concentrations *= alpha;

    Eigen::Vector<double, 20> newStationaryDistribution = sampleStationary(concentrations, gen);

    Eigen::Vector<double, 20> revConcentrations = newStationaryDistribution;
    revConcentrations *= alpha;

    double forward = Math::stationaryDirichletLogPDF(newStationaryDistribution, concentrations);
    double backward = Math::stationaryDirichletLogPDF(stationaryDistribution, revConcentrations);

    stationaryDistribution = newStationaryDistribution;
    double total = stationaryDistribution.sum();

    // To stop drift from summing up to 1.0
    for(int i = 0; i < 20; i++) {
        stationaryDistribution(i) = stationaryDistribution(i)/total;

        if(stationaryDistribution(i) < 1E-10) {
            return -1 * INFINITY;
        }
    }

    updated = true;

    return backward - forward;
}