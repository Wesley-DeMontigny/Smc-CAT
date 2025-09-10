#include "TransitionProbabilityClass.hpp"
#include <eigen3/Eigen/Eigenvalues>

TransitionProbabilityClass::TransitionProbabilityClass(int n, Eigen::Matrix<double, 20, 20>& bM) : baseMatrix(bM), updated(false) {
    // Initialize a buffer of transition probabilities for each branch
    for(int i = 0; i < n; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<double, 20, 20>::Zero() 
        );
    }
}

void TransitionProbabilityClass::recomputeEigens(){
    auto diagScaler = stationaryDistribution.asDiagonal();
    Eigen::Matrix<double, 20, 20> scaledMatrix = diagScaler * baseMatrix;

    for(int i = 0; i < 20; i++){
        double rowSum = 0.0;
        for(int j = 0; j < 20; j++){
            rowSum += scaledMatrix(i, j);
        }
        scaledMatrix(i,i) = -1.0 * rowSum;
    }

    Eigen::EigenSolver<Eigen::Matrix<double, 20, 20>> eigenSolver(scaledMatrix);
    eigenValues = eigenSolver.eigenvalues();
    eigenVectors = eigenSolver.eigenvectors();
    inverseEigenVectors = eigenVectors.inverse();
}

void TransitionProbabilityClass::recomputeTransitionProbs(int n, double t, double r){
    Eigen::DiagonalMatrix<std::complex<double>,20> diag;
    for(int i = 0; i < 20; i++)
        diag.diagonal()(i) = std::exp(eigenValues(i) * t * r);

    transitionProbabilities[n] = (eigenVectors * diag * inverseEigenVectors).real();
}