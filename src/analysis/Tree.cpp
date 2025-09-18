#include "Tree.hpp"
#include <memory>
#include <random>
#include <cassert>
#include <algorithm>

// Construct random tree with n tips and exponentially distributed branch lengths
Tree::Tree(int n){
    assert(n >= 3);

    auto generator = std::mt19937(std::random_device{}());

    std::exponential_distribution<double> branchDist(10.0);

    auto rootNode = std::unique_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}, false, false});
    root = rootNode.get();
    auto A = std::unique_ptr<TreeNode>(new TreeNode {0, "t0", true, false, branchDist(generator), root, {}, false, false});
    auto B = std::unique_ptr<TreeNode>(new TreeNode {1, "t1", true, false, branchDist(generator), root, {}, false, false});
    auto C = std::unique_ptr<TreeNode>(new TreeNode {2, "t2", true, false, branchDist(generator), root, {}, false, false});
    
    root->descendants.insert(A.get());
    root->descendants.insert(B.get());
    root->descendants.insert(C.get());

    tips.push_back(A.get());
    tips.push_back(B.get());
    tips.push_back(C.get());

    nodes.push_back(std::move(rootNode));
    nodes.push_back(std::move(A));
    nodes.push_back(std::move(B));
    nodes.push_back(std::move(C));

    for(int i = 3; i < n; i++){
        std::uniform_int_distribution<int> dist(0,i-1);
        int randomIndex = dist(generator);
        auto randomTip = tips[randomIndex];

        auto newInternal = std::unique_ptr<TreeNode>(new TreeNode {0, "", false, false, branchDist(generator), randomTip->ancestor, {}, false, false});
        auto newTip = std::unique_ptr<TreeNode>(new TreeNode {i, "t" + std::to_string(i), true, false, branchDist(generator), newInternal.get(), {}, false, false});
        tips.push_back(newTip.get());

        // Set the new internal node to be connected to the old ancestor and branch off with the two new nodes
        randomTip->ancestor = newInternal.get();
        newInternal->descendants.insert(newTip.get());
        newInternal->descendants.insert(randomTip);

        // Update old ancestor by adding a new descendant and removing the old one
        newInternal->ancestor->descendants.insert(newInternal.get());
        auto it = newInternal->ancestor->descendants.find(randomTip);
        if (it != newInternal->ancestor->descendants.end()) {
            newInternal->ancestor->descendants.erase(it);
        }

        nodes.push_back(std::move(newInternal));
        nodes.push_back(std::move(newTip));
    }

    // Assign node ID to the non-tips
    int counter = n;
    for(int i = 0; i < nodes.size(); i++){
        if(!nodes[i]->isTip){
            nodes[i]->id = counter;
            counter++;
        }
    }

    // Sort so we can access nodes by their ID
    std::sort(nodes.begin(), nodes.end(),
        [](const std::unique_ptr<TreeNode>& a, const std::unique_ptr<TreeNode>& b) {
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
    TreeNode*currentNode = nullptr;
    int counter = 0;

    for(auto character : s){
        if(character == '('){
            if(currentNode == nullptr){
                auto rootNode = std::unique_ptr<TreeNode>(new TreeNode {-1, "root", false, true, 0.0, nullptr, {}, false, false});
                currentNode = rootNode.get();
                root = rootNode.get();
                nodes.push_back(std::move(rootNode));
            }
            else {
                auto newInternal = std::unique_ptr<TreeNode>(new TreeNode {0, "", false, false, 0.0, currentNode, {}, false, false});
                currentNode->descendants.insert(newInternal.get());
                nodes.push_back(std::move(newInternal));
                currentNode = newInternal.get();
            }
        }
        else if(character == ':'){
            readingBranch = true;
            if(readingName){
                auto newTip = std::unique_ptr<TreeNode>(new TreeNode {counter, nameTokens, true, false, 0.0, currentNode, {}, false, false});
                currentNode->descendants.insert(newTip.get());
                currentNode = newTip.get();
                tips.push_back(newTip.get());
                nodes.push_back(std::move(newTip));

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

    for(int i = 0; i < nodes.size(); i++){
        if(!nodes[i]->isTip){
            nodes[i]->id = counter;
            counter++;
        }
    }

    // Sort so we can access nodes by their ID
    std::sort(nodes.begin(), nodes.end(),
        [](const std::unique_ptr<TreeNode>& a, const std::unique_ptr<TreeNode>& b) {
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

    root = nodes[t.root->id].get();

    for(int i = 0; i < nodes.size(); i++){
        auto p = nodes[i].get();
        auto q = t.nodes[i].get();

        p->id = q->id;
        p->isRoot = q->isRoot;
        p->isTip = q->isTip;
        p->name = q->name;
        p->updateCL = q->updateCL;
        p->updateTP = q->updateTP;
        p->branchLength = q->branchLength;
        p->descendants.clear();

        for(auto n : q->descendants)
            p->descendants.insert(nodes[n->id].get());

        if(q->ancestor != nullptr)
            p->ancestor = nodes[q->ancestor->id].get();
    }

    postOrder.clear();
    for(int i = 0; i < nodes.size(); i++){
        auto q = t.postOrder[i];
        postOrder.push_back(nodes[q->id].get());
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

void Tree::recursivePostOrderAssign(TreeNode* p){
    if(! p->isTip){
        for(auto child : p->descendants) {
            recursivePostOrderAssign(child);
        }
    }

    postOrder.push_back(p);
}

std::string Tree::recursiveNewickGenerate(std::string s, TreeNode* p){
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

double Tree::scaleBranchMove(double delta, std::mt19937& gen){
    std::uniform_real_distribution unifDist(0.0, 1.0);

    TreeNode* p = nullptr;
    do{
        p = nodes[(int)(unifDist(gen) * nodes.size())].get();
    }
    while(p == root);

    double currentV = p->branchLength;
    double scale = std::exp(delta * (unifDist(gen) - 0.5));
    double newV = currentV * scale;
    p->branchLength = newV;
    p->updateTP = true;

    TreeNode* q = p;
    do{
        q->updateCL = true;
        q = q->ancestor;
    } 
    while(q != root);
    root->updateCL = true;

    return std::log(scale);
}

int scaleSubtreeRecurse(TreeNode* n, double val){
    int c = 0;
    for(auto children : n->descendants){
        c += scaleSubtreeRecurse(children, val);
    }

    n->branchLength *= val;
    n->updateCL = true;
    n->updateTP = true;
    return c + 1;
}

double Tree::scaleSubtreeMove(double delta, std::mt19937& gen){
    std::uniform_real_distribution unifDist(0.0, 1.0);

    TreeNode* p = nullptr;
    do{
        p = nodes[(int)(unifDist(gen) * nodes.size())].get();
    }
    while(p == root);

    double scale = std::exp(delta * (unifDist(gen) - 0.5));
    int numBranches = scaleSubtreeRecurse(p, scale);

    TreeNode* q = p->ancestor;
    while(q != root){
        q->updateCL = true;
        q = q->ancestor;
    }
    root->updateCL = true;

    return numBranches * std::log(scale);
}

TreeNode* chooseNodeFromSet(const std::set<TreeNode*>& s, std::mt19937& gen){
    std::uniform_int_distribution<> dist(0, s.size() - 1);

    int index = dist(gen);

    auto it = s.begin();
    std::advance(it, index);

    return *it;
}

double Tree::NNIMove(std::mt19937& gen){
    std::uniform_real_distribution unifDist(0.0, 1.0);

    TreeNode* p = nullptr;
    do{
        p = nodes[(int)(unifDist(gen) * nodes.size())].get();
    }
    while(p == root || p->isTip);

    TreeNode* a = p->ancestor;

    std::set<TreeNode*> neighbors1 = p->descendants;
    TreeNode* n1 = chooseNodeFromSet(neighbors1, gen);

    // We exclude
    std::set<TreeNode*> neighbors2 = a->descendants;
    neighbors2.erase(p);
    TreeNode* n2 = chooseNodeFromSet(neighbors2, gen);

    n1->ancestor = a;
    n2->ancestor = p;
    a->descendants.insert(n1);
    p->descendants.insert(n2);
    
    auto it = p->descendants.find(n1);
    if (it != p->descendants.end()) {
        p->descendants.erase(it);
    }
    auto it2 = a->descendants.find(n2);
    if (it2 != a->descendants.end()) {
        a->descendants.erase(it2);
    }

    TreeNode* q = p;
    do{
        q->updateCL = true;
        q = q->ancestor;
    }
    while(q != root);
    root->updateCL = true;

    regeneratePostOrder();

    return 0.0;
}