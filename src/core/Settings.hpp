#ifndef SETTINGS_HPP
#define SETTINGS_HPP
#include <iostream>

/**
 * @brief This struct loads in the user's settings from the command line and provides
 * usage help. 
 */
struct Settings {
    #ifdef USE_UI
    Settings(void);
    #else
    Settings(void)=delete;
    #endif
    Settings(int argc, char* argv[]);

    void usage();

    int numParticles = 250;
    int rejuvenationIterations = 5;
    int numThreads = 10;
    bool invar = false;
    bool lg = false;
    double alg8Probability = 0.1;
    double rejuvenationThreshold = 0.6;
    int numRates = 1;
    unsigned int seed = 1;
    std::string fastaFile = "/workspaces/FastCAT/local_testing/globin_aa_aligned.fasta"; 
};

#endif