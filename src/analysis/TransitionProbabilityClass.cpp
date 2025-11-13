#include "TransitionProbabilityClass.hpp"
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions.hpp>
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

    stationaryDistribution = sampleStationary(rng, Eigen::Vector<double, 20>::Ones() * 5.0);
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

double TransitionProbabilityClass::lnPrior(){
    return TransitionProbabilityClass::stationarylnPdf(
        Eigen::Vector<double, 20>::Ones() * 5.0,
        stationaryDistribution
    );
}

double TransitionProbabilityClass::simplexMove(boost::random::mt19937& rng, double alpha){
    boost::random::uniform_01<double> unif{};
    double hastings = 0.0;

    int randomIndex = static_cast<int>(unif(rng) * 20);
    double currentValue = stationaryDistribution(randomIndex);

    double a = alpha + 1.0;
    double b = alpha / currentValue - a + 2.0;
    boost::random::beta_distribution<double> forwardDist{a, b};
    boost::math::beta_distribution<double> forwardDensity{a, b};
    double newValue = forwardDist(rng);
    stationaryDistribution(randomIndex) = newValue;

    double newB = alpha / newValue - a + 2.0;
    boost::math::beta_distribution<double> reverseDensity{a, newB};

    hastings += std::log(boost::math::pdf(reverseDensity, currentValue)) - std::log(boost::math::pdf(forwardDensity, newValue));

    double scaling = (1.0 - newValue) / (1.0 - currentValue);

    double sum = 0.0;
    for(int i = 0; i < 20; i++){
        if(i != randomIndex){
            stationaryDistribution(i) *= scaling;
        }

        if(stationaryDistribution(i) < 1e-30){
            return -INFINITY;
        }

        sum += stationaryDistribution(i);
    }

    stationaryDistribution /= sum;

    updated = true;

    hastings += 18 * std::log(scaling) - 19 * std::log(sum);

    return hastings;
}