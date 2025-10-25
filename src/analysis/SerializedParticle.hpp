#ifndef SERIALIZED_PARTICLE_HPP
#define SERIALIZED_PARTICLE_HPP
#include <eigen3/Eigen/Dense>
#include <vector>
/*
    The minimum information required to save the data from a model particle
*/
struct SerializedParticle {
    std::string newick;
    std::vector<int> assignments;
    std::vector<Eigen::Vector<double, 20>> stationaries;
    Eigen::Matrix<double, 20, 20> baseMatrix;
    double pInvar;
    double shape;
};

#endif