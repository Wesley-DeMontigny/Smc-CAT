#ifndef MCMC_HPP
#define MCMC_HPP
#include <boost/dynamic_bitset.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>

class Particle;

/**
 * @brief MCMC Functor
 */
class Mcmc {
    public:
        Mcmc(void);
        void operator()(Particle& m, int iterations, double tempering);
        void emplaceMove(std::tuple<double, std::function<int(Particle&)>, std::function<double(Particle&)>>&& move);
        void initMoveProbs();
    private:
        std::vector<std::tuple<double, std::function<int(Particle&)>, std::function<double(Particle&)>>> moves;
        std::vector<double> moveProbs;
        double totalMoveWeight;

};

#endif