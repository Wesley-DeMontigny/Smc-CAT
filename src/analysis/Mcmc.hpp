#ifndef MCMC_HPP
#define MCMC_HPP
#include <string>
#include <iostream>
#include <fstream>

class Particle;

/*

*/
class Mcmc {
    public:
        Mcmc(void);
        double run(Particle& m, int iterations, bool tune=true, bool debug=true, int tuneFrequency = 50, int printFrequency = 10, double tempering=1.0);
};

#endif