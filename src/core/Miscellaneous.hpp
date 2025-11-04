#ifndef MISC_HPP
#define MISC_HPP
#include <vector>
#include <iostream>
#include <boost/math/distributions.hpp>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <unordered_map>

#if MIXED_PRECISION
    using CL_TYPE = float;
#else
    using CL_TYPE = double;
#endif

/**
 * @brief Return the bins for a discretized gamma function
 */
inline void discretizeGamma(std::vector<double>& outVec, double shape, int num) {
    boost::math::gamma_distribution<double> gammaDist(shape, 1.0/shape);
    outVec.clear();
    outVec.reserve(num);

    double interval = 1.0 / (2.0 * num);
    for (int i = 0; i < num; ++i) {
        outVec.push_back(boost::math::quantile(gammaDist, (i * 2.0 + 1.0) * interval));
    }

    // Rescale categories
    double factor = num / std::accumulate(outVec.begin(), outVec.end(), 0.0);
    for (auto& v : outVec) {
        v *= factor;
    }
}

/**
 * @brief Compute the ESS, next temperature, and weights for SMC
 */
inline void computeNextStep(std::vector<double>& rawWeights, std::vector<double>& rawLogLikelihoods, 
                     std::vector<double>& normalizedWeights, double& ESS, const int& targetESS,
                     double& currentTemp, const double& lastTemp){
    int numParticles = rawWeights.size();
    double low = lastTemp;
    double high = 1.0;

    auto computeESS = [&](double temp){
        std::vector<double> tempWeights(numParticles);
        double total = 0.0;


        for(int n = 0; n < numParticles; n++){
            tempWeights[n] = rawWeights[n] + (temp - lastTemp) * rawLogLikelihoods[n];

        }

        double maxW = *std::max_element(tempWeights.begin(), tempWeights.end());

        std::vector<double> expWeights(numParticles);
        for(int n = 0; n < numParticles; n++){
            expWeights[n] = std::exp(tempWeights[n] - maxW);
            total += expWeights[n];
        }

        double sumSq = 0.0;
        for(int n = 0; n < numParticles; n++){
            expWeights[n] /= total;
            sumSq += expWeights[n] * expWeights[n];
        }

        return 1.0 / sumSq;
    };

    // Binary search for temperature
    while(high - low > 1e-5){
        double mid = 0.5 * (low + high);
        double midESS = computeESS(mid);

        if(midESS > targetESS){
            low = mid;   // We can still increase temperature
        } 
        else{
            high = mid;  // Too much degeneracy
        }
    }

    currentTemp = high;

    for (int n = 0; n < numParticles; n++){
        rawWeights[n] = rawWeights[n] + (currentTemp - lastTemp) * rawLogLikelihoods[n];
    }

    double maxW = *std::max_element(rawWeights.begin(), rawWeights.end());
    double total = 0.0;
    for (int n = 0; n < numParticles; n++){
        normalizedWeights[n] = std::exp(rawWeights[n] - maxW);
        total += normalizedWeights[n];
    }

    double sumSq = 0.0;
    for (int n = 0; n < numParticles; n++){
        normalizedWeights[n] /= total;
        sumSq += normalizedWeights[n] * normalizedWeights[n];
    }

    ESS = 1.0 / sumSq;
}

/**
 * @brief 
 * 
 */
inline void computeSplitPosteriors(std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities, 
                            const std::vector<double>& normalizedWeights,
                            const std::vector<std::set<boost::dynamic_bitset<>>>& particleSplits){
    splitPosteriorProbabilities.clear();
    for(int n = 0; n < particleSplits.size(); n++){
        for(boost::dynamic_bitset<> split : particleSplits[n]){
            if (splitPosteriorProbabilities.count(split)) {
                splitPosteriorProbabilities[split] += normalizedWeights[n];
            }
            else {
                splitPosteriorProbabilities[split] = normalizedWeights[n];
            }
        }
    }
}



#endif