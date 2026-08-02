#include "TransitionProbabilityClass.hpp"
#include "RateMatrices.hpp"
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions.hpp>
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>

inline double stableSigmoid(double x){
    if(x >= 0.0){
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(x);
    return e / (1.0 + e);
}

TransitionProbabilityClass::TransitionProbabilityClass(boost::random::mt19937& rng, int n, int c, Eigen::Vector<double, 190>* bM) : baseMatrix(bM), updated(false), numRates(c){
    // Initialize a buffer of transition probabilities for each branch
    transitionProbabilities.reserve(n * numRates);
    for(int i = 0; i < n * numRates; i++){
        transitionProbabilities.push_back(
            Eigen::Matrix<CL_TYPE, 20, 20>::Zero() 
        );
    }

    stationaryLogits = sampleStationaryLogits(rng);
    normalizeStationary();
}

Eigen::Vector<double, 20> TransitionProbabilityClass::sampleStationaryLogits(boost::random::mt19937& rng){
    // Draw from Dirichlet(2,...,2) via normalized Gamma(2,1), then map to centered stick-breaking coordinates.

    constexpr double eps = 1e-15;
    constexpr double dirichletAlpha = 2.0;
    Eigen::Vector<double, 20> simplex = Eigen::Vector<double, 20>::Zero();
    Eigen::Vector<double, 20> logits = Eigen::Vector<double, 20>::Zero();

    boost::random::gamma_distribution<double> gammaDist{dirichletAlpha, 1.0};
    double total = 0.0;
    for(int i = 0; i < 20; i++){
        simplex[i] = gammaDist(rng);
        total += simplex[i];
    }
    simplex /= total;

    double remainingStick = 1.0;
    for(int k = 0; k < 19; k++){
        double z = simplex[k] / remainingStick;
        z = std::max(eps, std::min(1.0 - eps, z));
        logits[k] = std::log(z) - std::log1p(-z) + std::log(static_cast<double>(20 - (k + 1)));
        remainingStick -= simplex[k];
        remainingStick = std::max(eps, remainingStick);
    }

    return logits;
}

void TransitionProbabilityClass::recomputeEigens(){
    Eigen::Matrix<double,20,20> Q = Eigen::Matrix<double,20,20>::Zero();
    const auto& coords = RateMatrices::contructLowerTriangleCoordinates();
    constexpr double eps = 1e-15;

    for(int c = 0; c < coords.size(); c++){
        const auto& [c1, c2] = coords[c];
        Q(c1,c2) = std::exp((*baseMatrix)(c));
        Q(c2,c1) = Q(c1, c2);
    }

    Q *= stationaryDistribution.asDiagonal();
    
    for (int i = 0; i < 20; i++){
        double offDiag = 0.0;
        for (int j = 0; j < 20; j++){
            if(j != i) offDiag += Q(i,j);
        }
        Q(i,i) = -offDiag;
    }

    // rescale mean rate to 1.0
    double meanRate = 0.0;
    for (int i = 0; i < 20; i++){
        meanRate += -Q(i,i) * stationaryDistribution(i);
    }
    Q /= meanRate;

    for(int i = 0; i < 20; i++){
        double piVal = std::max(stationaryDistribution[i], eps);
        sqrtPi[i] = std::sqrt(piVal);
        invSqrtPi[i] = 1.0 / sqrtPi[i];
    }

    Eigen::Matrix<double,20,20> S = sqrtPi.asDiagonal() * Q * invSqrtPi.asDiagonal();
    S = 0.5 * (S + S.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 20, 20>> eig(S);
    if(eig.info() != Eigen::Success){
        std::cout << "Failed symmetric eigendecomposition for CTMC transition construction." << std::endl;
        std::cout << "Stationary min/max: " << stationaryDistribution.minCoeff() << " / " << stationaryDistribution.maxCoeff() << std::endl;
        std::cout << "Q min/max: " << Q.minCoeff() << " / " << Q.maxCoeff() << std::endl;
        std::exit(1);
    }
    symEigenValues = eig.eigenvalues();
    symEigenVectors = eig.eigenvectors();
}

void TransitionProbabilityClass::recomputeTransitionProbs(int n, double t, int c, double r){
    constexpr double clampTol = 1e-12;
    constexpr double eps = 1e-15;
    double scaledTime = t * r;

    Eigen::Vector<double, 20> expEigen = (symEigenValues.array() * scaledTime).exp();
    Eigen::Matrix<double, 20, 20> M = symEigenVectors * expEigen.asDiagonal() * symEigenVectors.transpose();
    Eigen::Matrix<double, 20, 20> P = invSqrtPi.asDiagonal() * M * sqrtPi.asDiagonal();

    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 20; j++){
            if(std::abs(P(i,j)) < clampTol || P(i,j) < 0){
                P(i,j) = 0.0;
            }
        }
    }

    for(int i = 0; i < 20; i++){
        double rowSum = P.row(i).sum();
        if(rowSum <= eps || !std::isfinite(rowSum)){
            std::cout << "Invalid CTMC transition row sum after reversible expm." << std::endl;
            std::cout << "  Node: " << n << " Rate: " << c << std::endl;
            std::cout << "  Row: " << i << " RowSum: " << rowSum << std::endl;
            std::cout << "  T: " << t << " r: " << r << " Scaled: " << scaledTime << std::endl;
            std::cout << "  Min/Max P Row: " << P.row(i).minCoeff() << " / " << P.row(i).maxCoeff() << std::endl;
            std::exit(1);
        }
        P.row(i) /= rowSum;
    }

    #if MIXED_PRECISION
    transitionProbabilities[n*numRates + c] = P.cast<CL_TYPE>();
    #else
    transitionProbabilities[n*numRates + c] = P;
    #endif
}

double TransitionProbabilityClass::lnPrior(){
    // Symmetric Dirichlet(2) prior on simplex under centered stick-breaking transform.
    
    constexpr double eps = 1e-15;
    constexpr double dirichletAlpha = 2.0;

    double lnJ = 0.0;
    double lnDirichletKernel = 0.0;
    double remainingStick = 1.0;
    for(int k = 0; k < 19; k++){
        double centered = stationaryLogits[k] - std::log(static_cast<double>(20 - (k + 1)));
        double z = stableSigmoid(centered);
        z = std::max(eps, std::min(1.0 - eps, z));

        double simplexK = remainingStick * z;
        simplexK = std::max(eps, simplexK);
        lnDirichletKernel += (dirichletAlpha - 1.0) * std::log(simplexK);

        lnJ += std::log(remainingStick) + std::log(z) + std::log(1.0 - z);
        remainingStick *= (1.0 - z);
        remainingStick = std::max(eps, remainingStick);
    }
    lnDirichletKernel += (dirichletAlpha - 1.0) * std::log(remainingStick);

    double lnDirichletZ = std::lgamma(20.0 * dirichletAlpha) - 20.0 * std::lgamma(dirichletAlpha);

    return lnDirichletZ + lnDirichletKernel + lnJ;
}

void TransitionProbabilityClass::normalizeStationary(){
    // Centered stick-breaking transform from unconstrained space like Stan does
    constexpr double eps = 1e-15;

    double remainingStick = 1.0;
    for(int k = 0; k < 20 - 1; k++){
        double centered = stationaryLogits[k] - std::log(static_cast<double>(20 - (k + 1)));
        double z = stableSigmoid(centered);
        z = std::max(eps, std::min(1.0 - eps, z));

        stationaryDistribution[k] = remainingStick * z;
        remainingStick *= (1.0 - z);
    }
    stationaryDistribution[19] = std::max(eps, remainingStick);
    stationaryDistribution /= stationaryDistribution.sum();
}

double TransitionProbabilityClass::stationaryMove(boost::random::mt19937& rng, double delta){
    boost::random::uniform_01<double> unif{};

    // Only the first K-1 unconstrained coordinates parameterize simplex[K].
    int randomIndex = static_cast<int>(unif(rng) * 19);
    auto shiftDistribution = boost::random::normal_distribution<double>(0.0, delta);
    stationaryLogits(randomIndex) += shiftDistribution(rng);
    normalizeStationary();
    updated = true;

    return 0.0;
}
