#ifndef SETTINGS_HPP
#define SETTINGS_HPP
#include <iostream>

/*
    This struct loads in the user's settings from the command line and provides
    usage help.
*/
struct Settings {
    Settings(void) = delete;
    Settings(int argc, char* argv[]);
    std::string alignment_file;
    std::string mcmc_trace = "trace.log";
    std::string tree_trace = "trees.log";
    std::string starting_tree;

    int num_iter = 1000;
    int sampling_freq = 100;
};

#endif