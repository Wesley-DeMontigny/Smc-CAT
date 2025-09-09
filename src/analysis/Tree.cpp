#include "Tree.hpp"
#include <memory>
#include <random>
#include <cassert>

// Construct random tree with n tips and exponentially distributed branch lengths
Tree::Tree(int n){
    assert(n >= 3);

    auto generator = std::mt19937(std::random_device{}());

    std::exponential_distribution<double> branchDist(0.5);

    root = std::shared_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}});
    auto A = std::shared_ptr<TreeNode>(new TreeNode {0, "t1", true, false, branchDist(generator), root, {}});
    auto B = std::shared_ptr<TreeNode>(new TreeNode {1, "t2", true, false, branchDist(generator), root, {}});
    auto C = std::shared_ptr<TreeNode>(new TreeNode {2, "t3", true, false, branchDist(generator), root, {}});

    tips.push_back(A);
    tips.push_back(B);
    tips.push_back(C);

    for(int i = 3; i < n; i++){
        std::uniform_int_distribution<int> dist(0,i-1);
        int randomIndex = dist(generator);
        auto randomTip = tips[randomIndex];

        auto newTip = std::shared_ptr<TreeNode>(new TreeNode {i, "t" + std::to_string(i+1), true, false, branchDist(generator), nullptr, {}});
        tips.push_back(newTip);
        auto newInternal = std::shared_ptr<TreeNode>(new TreeNode {0, "", false, false, branchDist(generator), nullptr, {}});

        // Set the new internal node to be connected to the old ancestor and branch off with the two new nodes
        newInternal->ancestor = randomTip->ancestor;
        randomTip->ancestor = newInternal;
        newTip->ancestor = newInternal;
        newInternal->descendants.insert(newTip);
        newInternal->descendants.insert(randomTip);

        // Update old ancestor by adding a new descendant and removing the old one
        newInternal->ancestor->descendants.insert(newInternal);
        auto it = newInternal->ancestor->descendants.find(randomTip);
        if (it != newInternal->ancestor->descendants.end()) {
            newInternal->ancestor->descendants.erase(it);
        }
    }

    // Assign node ID to the non-tips
    int counter = n;
    recursiveIDAssign(n, root);

    regeneratePostOrder();
}

// Construct from newick
Tree::Tree(std::string s){

}

Tree::~Tree() {}

std::string Tree::generateNewick(){
    std::string output = "";

    output = recursiveNewickGenerate(output, root);

    output += ";";

    return output;
}

void Tree::regeneratePostOrder(){
    postOrder.clear();

    recursivePostOrderAssign(root);
}

int Tree::recursiveIDAssign(int n, std::shared_ptr<TreeNode> p){
    for(auto child : p->descendants) {
        if(! p->isTip){
            n = recursiveIDAssign(n, child);
        }
    }

    p->id = n;
    n++;

    return n;
}

void Tree::recursivePostOrderAssign(std::shared_ptr<TreeNode> p){
    if(! p->isTip){
        for(auto child : p->descendants) {
            recursivePostOrderAssign(child);
        }
    }

    postOrder.push_back(p);
}

std::string Tree::recursiveNewickGenerate(std::string s, std::shared_ptr<TreeNode> p){
    if(! p->isTip){
        s += "(";
        for(auto child : p->descendants) {
            s = recursiveNewickGenerate(s, child);
            s += ",";
        }
        s.pop_back();
        s += ")";

        if(p != root){
            s += ":" + std::to_string(p->branchLength);
        }
    }
    else {
        s += p->name + ":" + std::to_string(p->branchLength);
    }

    return s;
}