#ifndef SERIALIZED_PARTICLE_HPP
#define SERIALIZED_PARTICLE_HPP
#include <eigen3/Eigen/Dense>
#include <vector>
/*
    The minimum information required to save the data from a model particle
*/
struct SerializedParticle {
    std::string newick;
    std::vector<double> branchLengths; // This is redundant with the Newick, but allows us to tune proposals
    std::vector<int> assignments;
    std::vector<int> categorySize;
    std::vector<Eigen::Vector<double, 20>> stationaries;
    Eigen::Vector<double, 190> baseMatrix;
    double pInvar;
    double shape;
};

#endif