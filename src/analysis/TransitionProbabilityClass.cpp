#include "TransitionProbabilityClass.hpp"
#include <boost/random/gamma_distribution.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>

TransitionProbabilityClass::TransitionProbabilityClass(boost::random::mt19937& rng, int n, int c, Eigen::Matrix<double, 20, 20>* bM) : baseMatrix(bM), updated(false), numRates(c) {
    // Initialize a buffer of transition probabilities for each branch
    transitionProbabilities.reserve(n * numRates);
    for(int i = 0; i < n * numRates; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<CL_TYPE, 20, 20>::Zero() 
        );
    }

    workingMatrix1 = Eigen::Matrix<Eigen::dcomplex, 20, 20>::Zero();
    workingMatrix2 = Eigen::Matrix<Eigen::dcomplex, 20, 20>::Zero();
    workingDiag = Eigen::DiagonalMatrix<Eigen::dcomplex, 20>{};

    stationaryDistribution = sampleStationary(rng, Eigen::Vector<double, 20>::Ones());
}

Eigen::Vector<double, 20> TransitionProbabilityClass::sampleStationary(boost::random::mt19937& rng, const Eigen::Vector<double, 20>& alpha){
    Eigen::Vector<double, 20> stationaryDistribution = Eigen::Vector<double, 20>::Zero();

    double denom = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        double randGamma = boost::random::gamma_distribution<double>{alpha(i), 1.0}(rng);
        stationaryDistribution(i) = randGamma;
        denom += randGamma;
    }

    return stationaryDistribution / denom;
}

double TransitionProbabilityClass::stationarylnPdf(const Eigen::Vector<double, 20>& alpha, 
                                                   const Eigen::Vector<double, 20>& x) {
    double alpha0 = alpha.sum();
    double lnPdf = std::lgamma(alpha0);

    for(int i = 0; i < 20; i++){
        lnPdf -= std::lgamma(alpha[i]);
        lnPdf += (alpha[i] - 1.0) * std::log(x[i]);
    }
    return lnPdf;
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
    for(int i = 0; i < 20; i++)
        workingDiag.diagonal()(i) = std::exp(eigenValues(i) * t * r);

    workingMatrix1.noalias() = eigenVectors * workingDiag;
    workingMatrix2.noalias() = workingMatrix1 * inverseEigenVectors;

    #if MIXED_PRECISION
    transitionProbabilities[n*numRates + c] = workingMatrix2.real().cast<CL_TYPE>();
    #else
    transitionProbabilities[n*numRates + c] = workingMatrix2.real();
    #endif
}

double TransitionProbabilityClass::dirichletSimplexMove(boost::random::mt19937& rng, double alpha){
    Eigen::Vector<double, 20> concentrations = stationaryDistribution;
    concentrations *= alpha;

    Eigen::Vector<double, 20> newStationaryDistribution = sampleStationary(rng, concentrations);

    Eigen::Vector<double, 20> revConcentrations = newStationaryDistribution;
    revConcentrations *= alpha;

    double forward = stationarylnPdf(concentrations, newStationaryDistribution);
    double backward = stationarylnPdf(revConcentrations, stationaryDistribution);

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