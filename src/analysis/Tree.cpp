#include "Tree.hpp"
#include <memory>
#include <random>
#include <cassert>
#include <algorithm>

// Construct random tree with n tips and exponentially distributed branch lengths
Tree::Tree(int n){
    assert(n >= 3);

    auto generator = std::mt19937(std::random_device{}());

    std::exponential_distribution<double> branchDist(2.0);

    root = std::shared_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}, false, false});
    auto A = std::shared_ptr<TreeNode>(new TreeNode {0, "t0", true, false, branchDist(generator), root, {}, false, false});
    auto B = std::shared_ptr<TreeNode>(new TreeNode {1, "t1", true, false, branchDist(generator), root, {}, false, false});
    auto C = std::shared_ptr<TreeNode>(new TreeNode {2, "t2", true, false, branchDist(generator), root, {}, false, false});
    
    root->descendants.insert(A);
    root->descendants.insert(B);
    root->descendants.insert(C);
    
    nodes.push_back(root);
    nodes.push_back(A);
    nodes.push_back(B);
    nodes.push_back(C);

    tips.push_back(A);
    tips.push_back(B);
    tips.push_back(C);

    for(int i = 3; i < n; i++){
        std::uniform_int_distribution<int> dist(0,i-1);
        int randomIndex = dist(generator);
        auto randomTip = tips[randomIndex];

        auto newInternal = std::shared_ptr<TreeNode>(new TreeNode {0, "", false, false, branchDist(generator), randomTip->ancestor, {}, false, false});
        auto newTip = std::shared_ptr<TreeNode>(new TreeNode {i, "t" + std::to_string(i), true, false, branchDist(generator), newInternal, {}, false, false});
        tips.push_back(newTip);
        nodes.push_back(newInternal);
        nodes.push_back(newTip);

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
    for(auto node : nodes){
        if(!node->isTip){
            node->id = counter;
            counter++;
        }
    }

    // Sort so we can access nodes by their ID
    std::sort(nodes.begin(), nodes.end(),
        [](const std::shared_ptr<TreeNode>& a, const std::shared_ptr<TreeNode>& b) {
            return a->id < b->id;
        }
    );

    regeneratePostOrder();
}

// Construct from newick
Tree::Tree(std::string s){

    auto generator = std::mt19937(std::random_device{}());

    std::exponential_distribution<double> branchDist(2.0);

    std::string nameTokens = "";
    std::string branchTokens = "";
    bool readingName = false;
    bool readingBranch = false;
    std::shared_ptr<TreeNode> currentNode = nullptr;
    int counter = 0;

    for(auto character : s){
        if(character == '('){
            if(currentNode == nullptr){
                root = std::shared_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}, false, false});
                currentNode = root;
                nodes.push_back(root);
            }
            else {
                auto newInternal = std::shared_ptr<TreeNode>(new TreeNode {0, "", false, false, 0.0, currentNode, {}, false, false});
                currentNode->descendants.insert(newInternal);
                nodes.push_back(newInternal);
                currentNode = newInternal;
            }
        }
        else if(character == ':'){
            readingBranch = true;
            if(readingName){
                auto newTip = std::shared_ptr<TreeNode>(new TreeNode {counter, nameTokens, true, false, 0.0, currentNode, {}, false, false});
                currentNode->descendants.insert(newTip);
                currentNode = newTip;
                tips.push_back(newTip);
                nodes.push_back(newTip);

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

    for(auto node : nodes){
        if(!node->isTip){
            node->id = counter;
            counter++;
        }
    }

    // Sort so we can access nodes by their ID
    std::sort(nodes.begin(), nodes.end(),
        [](const std::shared_ptr<TreeNode>& a, const std::shared_ptr<TreeNode>& b) {
            return a->id < b->id;
        }
    );

    regeneratePostOrder();

    for(auto node : postOrder){
        if(node != root){
            if(node->branchLength <= 0.0){
                node->branchLength = branchDist(generator); // If we aren't provided a branch length make one up
            }
        }
    }
}

Tree::Tree(std::vector<std::string> tList) : Tree(tList.size()){
    for(int i = 0; i < tips.size(); i++){
        tips[i]->name = tList[i];
    }
}

Tree::Tree(const Tree& t) : Tree(t.tips.size()){
    clone(t);
}

Tree& Tree::operator=(const Tree& t) {
    if(this == &t)
        return *this;
    
    clone(t);
    return *this;
}

void Tree::clone(const Tree& t){
    assert(nodes.size() == t.nodes.size());

    root = nodes[t.root->id];

    for(int i = 0; i < nodes.size(); i++){
        auto p = nodes[i];
        auto q = t.nodes[i];

        p->id = q->id;
        p->isRoot = q->isRoot;
        p->isTip = q->isTip;
        p->name = q->name;
        p->updateCL = q->updateCL;
        p->updateTP = q->updateTP;
        p->branchLength = q->branchLength;
        p->descendants.clear();

        for(auto n : q->descendants)
            p->descendants.insert(nodes[n->id]);

        if(q->ancestor != nullptr)
            p->ancestor = nodes[q->ancestor->id];
    }

    postOrder.clear();
    for(int i = 0; i < nodes.size(); i++){
        auto q = t.postOrder[i];
        postOrder.push_back(nodes[q->id]);
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

void Tree::updateAll(){
    for(auto n : postOrder){
        n->updateCL = true;
        n->updateTP = true;
    }
}

double Tree::localMove(double delta){

}

double Tree::scaleBranchMove(double delta){

}

double Tree::NNI(){

}