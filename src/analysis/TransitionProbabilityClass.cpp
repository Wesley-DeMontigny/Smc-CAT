#include "TransitionProbabilityClass.hpp"
#include <eigen3/Eigen/Eigenvalues>
#include <random>

TransitionProbabilityClass::TransitionProbabilityClass(int n, std::shared_ptr<Eigen::Matrix<double, 20, 20>> bM) : baseMatrix(bM), updated(false) {
    // Initialize a buffer of transition probabilities for each branch
    for(int i = 0; i < n; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<double, 20, 20>::Zero() 
        );
    }

    transitionProbabilities.shrink_to_fit();

    // Sample from a dirichlet
    auto generator = std::mt19937(std::random_device{}());

    stationaryDistribution = Eigen::Vector<double, 20>::Zero();
    double denom = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        std::gamma_distribution<double> gammaDist(1.5, 1.0); // Set alpha to be 1.5 just to get a roughly centered initialization
        double randGamma = gammaDist(generator);
        stationaryDistribution[i] = randGamma;
        denom += randGamma;
    }
    stationaryDistribution /= denom;
}

void TransitionProbabilityClass::recomputeEigens(){
    auto diagScaler = stationaryDistribution.asDiagonal();
    Eigen::Matrix<double, 20, 20> scaledMatrix = diagScaler * (*baseMatrix);

    double rescaleFactor = 0.0;
    for(int i = 0; i < 20; i++){
        double rowSum = 0.0;
        for(int j = 0; j < 20; j++){
            rowSum += scaledMatrix(i, j);
        }
        scaledMatrix(i,i) = -1.0 * rowSum;
        rescaleFactor += rowSum;
    }

    scaledMatrix /= rescaleFactor; // Force out-rate sum up to 1.0

    Eigen::EigenSolver<Eigen::Matrix<double, 20, 20>> eigenSolver(scaledMatrix);
    eigenValues = eigenSolver.eigenvalues();
    eigenVectors = eigenSolver.eigenvectors();
    inverseEigenVectors = eigenVectors.inverse();
}

void TransitionProbabilityClass::recomputeTransitionProbs(int n, double t, double r){
    Eigen::DiagonalMatrix<std::complex<double>,20> diag;
    for(int i = 0; i < 20; i++)
        diag.diagonal()(i) = std::exp(eigenValues(i) * t * r);

    auto transProb = (eigenVectors * diag * inverseEigenVectors).real().transpose(); // We transpose it so we can do A * v

    transitionProbabilities[n] = transProb;
}