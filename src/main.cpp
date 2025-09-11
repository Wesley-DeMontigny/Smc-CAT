#include <iostream>
#include "core/Settings.hpp"
#include "analysis/RateMatrices.hpp"
#include "analysis/Model.hpp"
#include "core/Alignment.hpp"

int main() {
    Alignment aln("/workspaces/FastCAT/local_testing/globin_test.fasta");
    Model m(aln);
    
    return 0;
}