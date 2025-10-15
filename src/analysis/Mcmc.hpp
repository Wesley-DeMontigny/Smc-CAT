#ifndef MCMC_HPP
#define MCMC_HPP
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>

class Particle;

/*

*/
class Mcmc {
    public:
        Mcmc(void);
        double run(Particle& m, int iterations, const std::unordered_map<std::string, double>& splitPosterior, bool updateInvar, double tempering);
};

#endif