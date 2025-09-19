#include "TransitionProbabilityClass.hpp"
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>

double stationaryDirichletLogPDF(const Eigen::Vector<double, 20>& x,
                                 const Eigen::Vector<double, 20>& alpha) {
    double alpha0 = alpha.sum();
    double lnPdf = std::lgamma(alpha0);

    for(int i = 0; i < 20; i++){
        lnPdf -= std::lgamma(alpha[i]);
        lnPdf += (alpha[i] - 1.0) * std::log(x[i]);
    }
    return lnPdf;
}

TransitionProbabilityClass::TransitionProbabilityClass(int n, Eigen::Matrix<double, 20, 20>* bM) : baseMatrix(bM), updated(false) {
    // Initialize a buffer of transition probabilities for each branch
    for(int i = 0; i < n; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<double, 20, 20>::Zero() 
        );
    }

    transitionProbabilities.shrink_to_fit();

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

void TransitionProbabilityClass::recomputeTransitionProbs(int n, double t, double r){
    Eigen::DiagonalMatrix<std::complex<double>,20> diag;
    for(int i = 0; i < 20; i++)
        diag.diagonal()(i) = std::exp(eigenValues(i) * t * r);

    auto transProb = (eigenVectors * diag * inverseEigenVectors).real();

    transitionProbabilities[n] = transProb;
}

double TransitionProbabilityClass::dirichletSimplexMove(double alpha, std::mt19937& gen){
    Eigen::Vector<double, 20> concentrations = stationaryDistribution;
    concentrations *= alpha;

    Eigen::Vector<double, 20> newStationaryDistribution = sampleStationary(concentrations, gen);

    Eigen::Vector<double, 20> revConcentrations = newStationaryDistribution;
    revConcentrations *= alpha;

    double forward = stationaryDirichletLogPDF(newStationaryDistribution, concentrations);
    double backward = stationaryDirichletLogPDF(stationaryDistribution, revConcentrations);

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