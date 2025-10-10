#ifndef SERIALIZED_PARTICLE_HPP
#define SERIALIZED_PARTICLE_HPP
#include <vector>
#include <eigen3/Eigen/Dense>

/*
    The minimum information required to save the data from a model particle
*/
struct SerializedParticle {
    SerializedParticle(void)=delete;

    std::string newick;
    std::vector<int> assignments;
    std::vector<Eigen::Vector<double, 20>> stationaries;
    double pInvar;
    double shape;
};

#endif