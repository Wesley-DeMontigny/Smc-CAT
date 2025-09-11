#ifndef PROBABILITY_FUNCTIONS_HPP
#include<cmath>

namespace ProbabilityFunctions {
    double GammaLogPDF(double x, double shape, double rate){
        if (x <= 0.0) return -INFINITY; 
        return shape * std::log(rate) + (shape - 1.0) * std::log(x) - rate * x - std::lgamma(shape);
    }
}

#endif