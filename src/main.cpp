#include <iostream>
#include "core/Settings.hpp"
#include "core/Alignment.hpp"
#include "analysis/Tree.hpp"

int main() {
    Tree myTree(5);
    std::string newick = myTree.generateNewick();
    std::cout << newick << std::endl;
    Tree newickTree(newick);
    std::cout << newickTree.generateNewick() << std::endl;

    return 0;
}