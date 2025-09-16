#include <iostream>
#include "core/Settings.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/Model.hpp"
#include "core/Alignment.hpp"
#include "analysis/Mcmc.hpp"
#include <random>

int main() {
    Alignment aln("/workspaces/FastCAT/local_testing/globin_test.fasta");
    Model m(aln);
    Mcmc analysis(aln, m);

    analysis.burnin(10000);
    
    return 0;
}