#include "TransitionProbabilityClass.hpp"
#include <eigen3/Eigen/Eigenvalues>

Eigen::Vector<double, 20> sampleStationaryDist(Eigen::Vector<double, 20> alpha, std::mt19937& gen){
    Eigen::Vector<double, 20> stationaryDistribution = Eigen::Vector<double, 20>::Zero();

    double denom = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        std::gamma_distribution<double> gammaDist(alpha[i], 1.0);
        double randGamma = gammaDist(gen);
        stationaryDistribution[i] = randGamma;
        denom += randGamma;
    }

    return stationaryDistribution / denom;
}

double stationaryDirichletLogPDF(Eigen::Vector<double, 20> x, Eigen::Vector<double, 20> alpha){
    double alpha0 = alpha.sum();

    double lnPdf = -std::lgamma(alpha0);
    for(int i = 0; i < 20; i++){
        lnPdf += std::lgamma(alpha[i]);
        lnPdf += x[i] * (alpha[i] - 1.0);
    }

    return lnPdf;
}

TransitionProbabilityClass::TransitionProbabilityClass(int n, std::shared_ptr<Eigen::Matrix<double, 20, 20>> bM) : baseMatrix(bM), updated(false) {
    // Initialize a buffer of transition probabilities for each branch
    for(int i = 0; i < n; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<double, 20, 20>::Zero() 
        );
    }

    transitionProbabilities.shrink_to_fit();

    // Sample from a dirichlet with very mild centering
    auto generator = std::mt19937(std::random_device{}());
    stationaryDistribution = sampleStationaryDist(Eigen::Vector<double, 20>::Ones() * 1.5, generator);
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

double TransitionProbabilityClass::dirichletSimplexMove(double alpha, double offset, std::mt19937& gen){
    Eigen::Vector<double, 20> concentrations = stationaryDistribution;
    for(int i = 0; i < 20; i++){
        concentrations[i] += offset;
    }
    concentrations *= alpha;

    Eigen::Vector<double, 20> newStationaryDistribution = sampleStationaryDist(concentrations, gen);

    Eigen::Vector<double, 20> revConcentrations = newStationaryDistribution;
    for(int i = 0; i < 20; i++){
        revConcentrations[i] += offset;
    }
    revConcentrations *= alpha;

    double forward = stationaryDirichletLogPDF(stationaryDistribution, revConcentrations);
    double backward = stationaryDirichletLogPDF(newStationaryDistribution, concentrations);

    updated = true;

    return forward - backward;
}