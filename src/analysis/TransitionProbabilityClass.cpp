#include "TransitionProbabilityClass.hpp"
#include "RateMatrices.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>

TransitionProbabilityClass::TransitionProbabilityClass(boost::random::mt19937& rng, int n, int c, Eigen::Vector<double, 190>* bM) : baseMatrix(bM), updated(false), numRates(c) {
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

    stationaryLogits = sampleStationaryLogits(rng);
    normalizeStationary();
}

Eigen::Vector<double, 20> TransitionProbabilityClass::sampleStationaryLogits(boost::random::mt19937& rng){
    Eigen::Vector<double, 20> stationaryDistribution = Eigen::Vector<double, 20>::Zero();

    for (size_t i = 0; i < 19; ++i) {
        double randLogit = boost::random::normal_distribution<double>{0.0, 1.0}(rng);
        stationaryDistribution(i) = randLogit;
    }

    return stationaryDistribution;
}

/**
 * TODO: Optionally allow for dirichlet prior. The mapping from logit to simplex has a log jacobian determinant factor of sum(log(x_i)) where x_i is the simplex value
 */
double TransitionProbabilityClass::stationarylnPdf(const Eigen::Vector<double, 20>& x) {
    boost::math::normal_distribution<double> standard_normal{};

    double lnPdf = 0.0;
    for(const auto& z : x)
        lnPdf += std::log(boost::math::pdf(standard_normal, z));

    return lnPdf;
}

void TransitionProbabilityClass::recomputeEigens(){
    Eigen::Matrix<double,20,20> Q = Eigen::Matrix<double,20,20>::Zero();
    const auto& coords = RateMatrices::contructLowerTriangleCoordinates();

    for(int c = 0; c < coords.size(); c++){
        const auto& [c1, c2] = coords[c];
        Q(c1,c2) = std::exp((*baseMatrix)(c));
        Q(c2,c1) = Q(c1, c2);
    }

    Q *= stationaryDistribution.asDiagonal();
    
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

double TransitionProbabilityClass::lnPrior(){
    return TransitionProbabilityClass::stationarylnPdf(stationaryLogits);
}

void TransitionProbabilityClass::normalizeStationary(){
    double totalExp = 0.0;
    for(int i = 0; i < 20; i++){
        double newExp = std::exp(stationaryLogits[i]);
        totalExp += newExp;
        stationaryDistribution[i] = newExp;
    }
    stationaryDistribution /= totalExp;
}

double TransitionProbabilityClass::stationaryMove(boost::random::mt19937& rng, double delta){
    boost::random::uniform_01<double> unif{};

    int randomIndex = static_cast<int>(unif(rng) * 20);
    double currentValue = stationaryDistribution(randomIndex);

    auto shiftDistribution = boost::random::normal_distribution<double>(0.0, delta);
    auto proposalDensity = boost::math::normal_distribution<double>(0.0, delta);
    
    double newValue = shiftDistribution(rng) + currentValue;

    stationaryDistribution(randomIndex) = newValue;
    normalizeStationary();
    updated = true;

    return 0.0;
}