#include <iostream>
#include "core/Settings.hpp"
#include "core/Alignment.hpp"
#include "analysis/Tree.hpp"

int main() {
    Tree myTree(10);

    std::cout << myTree.generateNewick() << std::endl;

    return 0;
}