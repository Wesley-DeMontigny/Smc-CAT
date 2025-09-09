#include "Tree.hpp"
#include <memory>
#include <random>
#include <cassert>
#include <iostream>

// Construct random tree with n tips and exponentially distributed branch lengths
Tree::Tree(int n){
    assert(n >= 3);

    auto generator = std::mt19937(std::random_device{}());

    std::exponential_distribution<double> branchDist(2.0);

    root = std::shared_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}});
    auto A = std::shared_ptr<TreeNode>(new TreeNode {0, "t0", true, false, branchDist(generator), root, {}});
    auto B = std::shared_ptr<TreeNode>(new TreeNode {1, "t1", true, false, branchDist(generator), root, {}});
    auto C = std::shared_ptr<TreeNode>(new TreeNode {2, "t2", true, false, branchDist(generator), root, {}});
    
    root->descendants.insert(A);
    root->descendants.insert(B);
    root->descendants.insert(C);

    tips.push_back(A);
    tips.push_back(B);
    tips.push_back(C);

    for(int i = 3; i < n; i++){
        std::uniform_int_distribution<int> dist(0,i-1);
        int randomIndex = dist(generator);
        auto randomTip = tips[randomIndex];

        auto newInternal = std::shared_ptr<TreeNode>(new TreeNode {0, "", false, false, branchDist(generator), randomTip->ancestor, {}});
        auto newTip = std::shared_ptr<TreeNode>(new TreeNode {i, "t" + std::to_string(i), true, false, branchDist(generator), newInternal, {}});
        tips.push_back(newTip);

        // Set the new internal node to be connected to the old ancestor and branch off with the two new nodes
        randomTip->ancestor = newInternal;
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

    std::string nameTokens = "";
    std::string branchTokens = "";
    bool readingName = false;
    bool readingBranch = false;
    std::shared_ptr<TreeNode> currentNode = nullptr;
    int counter = 0;

    for(auto character : s){
        if(character == '('){
            if(currentNode == nullptr){
                root = std::shared_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}});
                currentNode = root;
            }
            else {
                auto newInternal = std::shared_ptr<TreeNode>(new TreeNode {0, "", false, false, 0.0, currentNode, {}});
                currentNode->descendants.insert(newInternal);
                currentNode = newInternal;
            }
        }
        else if(character == ':'){
            readingBranch = true;
            if(readingName){
                auto newTip = std::shared_ptr<TreeNode>(new TreeNode {counter, nameTokens, true, false, 0.0, currentNode, {}});
                currentNode->descendants.insert(newTip);
                currentNode = newTip;
                tips.push_back(newTip);

                counter++;
                nameTokens = "";
                readingName = false;
            }
        }
        else if(character == ')' || character == ','){
            if(readingBranch){
                currentNode->branchLength = std::stod(branchTokens);
                branchTokens = "";
                readingBranch = false;
            }

            currentNode = currentNode->ancestor;
        }
        else if(character == ';'){
            break;
        }
        else {
            if(readingBranch){
                branchTokens += character;
            }
            else{
                readingName = true;
                nameTokens += character;
            }
        }
    }

    recursiveIDAssign(counter, root);

    regeneratePostOrder();

    for(auto node : postOrder){
        if(node != root){
            assert(node->branchLength > 0.0);
        }
    }
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