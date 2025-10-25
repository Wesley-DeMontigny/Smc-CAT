#ifndef MISC_HPP
#define MISC_HPP
#include <vector>
#include <iostream>
#include <boost/math/distributions.hpp>

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


#endif