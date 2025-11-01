#ifndef SETTINGS_HPP
#define SETTINGS_HPP
#include <iostream>

/**
 * @brief This struct loads in the user's settings from the command line and provides
 * usage help. 
 */
struct Settings {
    Settings(void);
    Settings(int argc, char* argv[]);

    void usage();

    int numParticles = 250;
    int rejuvenationIterations = 10;
    int numThreads = 1;
    bool invar = false;
    bool lg = false;
    double alg8Probability = 0.1;
    int numRates = 1;
    unsigned int seed = 1;
    std::string fastaFile; 
};

#endif