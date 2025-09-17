#ifndef MCMC_HPP
#define MCMC_HPP
#include <string>
#include <iostream>
#include <fstream>

class Alignment;
class Model;

/*

*/
class Mcmc {
    public:
        Mcmc(void)=delete;
        Mcmc(Alignment& aln, Model& m);

        void burnin(int iterations, int tuneFrequency, int printFrequency);
        void run(int iterations);
    private:
        Alignment& alignment;
        Model& model;

        std::string analysisLog;
        std::string treeLog;
};

#endif