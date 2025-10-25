#ifndef MCMC_HPP
#define MCMC_HPP
#include <boost/dynamic_bitset.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>

class Particle;

/**
 * @brief 
 * 
 */
class Mcmc {
    public:
        Mcmc(void);
        double run(Particle& m, int iterations, const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosterior, bool updateInvar, double tempering);
};

#endif