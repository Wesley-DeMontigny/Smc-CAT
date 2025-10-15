#include "Tree.hpp"
#include <memory>
#include <random>
#include <cassert>
#include <algorithm>
#include <iostream>

// Construct random tree with n tips and exponentially distributed branch lengths
Tree::Tree(int n){
    assert(n >= 3);

    auto generator = std::mt19937(std::random_device{}());

    root = addNode();
    root->isRoot = true;
    root->branchLength = 0.0;
    auto A = addNode();
    A->isTip = true;
    A->ancestor = root;
    A->name = "t0";
    A->id = 0;
    auto B = addNode();
    B->isTip = true;
    B->ancestor = root;
    B->name = "t1";
    B->id = 1;
    
    root->descendants.insert(A);
    root->descendants.insert(B);

    tips.push_back(A);
    tips.push_back(B);

    for(int i = 2; i < n; i++){
        std::uniform_int_distribution<int> dist(0,i-1);
        int randomIndex = dist(generator);
        auto randomTip = tips[randomIndex];

        auto newInternal = addNode();
        newInternal->ancestor = randomTip->ancestor;
        auto newTip = addNode();
        newTip->id = i;
        newTip->isTip = true;
        newTip->name = "t" + std::to_string(i);
        newTip->ancestor = newInternal;
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
    int id = tips.size();
    for(auto& n : nodes){
        if(n->isTip == false){
            n->id = id++;
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

Tree::Tree(std::string newick, std::vector<std::string> taxaNames){
    std::vector<std::string> tokens = parseNewickString(newick);

    TreeNode* p = nullptr;
    bool readingBranchLength = false;

    for(std::string tok : tokens){
        if(tok == "("){
            TreeNode* newNode = addNode();
            if(p == nullptr){
                root = newNode;
                newNode->isRoot = true;
            }
            else{
                p->descendants.insert(newNode);
                newNode->ancestor = p;
            }

            p = newNode;
        }
        else if(tok == ")" || tok == ","){
            if(p->ancestor != nullptr)
                p = p->ancestor;
            else{
                std::cerr << "Poorly formatted Newick! Non-root node has no ancestor!" << std::endl;
                std::exit(1);
            }
        }
        else if(tok == ":"){
            readingBranchLength = true;
        }
        else if(tok == ";"){
            if(p != root){
                std::cerr << "Poorly formatted Newick! Did not end at root." << std::endl;
                std::exit(1);
            }
        }
        else{
            if(readingBranchLength){
                double x = atof(tok.c_str());
                p->branchLength = x;
            }
            else{
                //We need to trim the white space at the beginning and end of the token
                while(tok[0] == ' ')
                    tok.erase(0,1);
                while(tok[tok.size()-1] == ' ')
                    tok.erase(tok.size()-1);

                // There is a weird bug causing spaces elsewhere to not be parsed correctly? I am putting this here to fix that
                if(tok != ""){
                    TreeNode* newNode = addNode();
                    newNode->ancestor = p;
                    p->descendants.insert(newNode);
                    newNode->name = tok;
                    newNode->isTip = true;
                    tips.push_back(newNode);

                    int taxonIndex = getTaxonIndex(tok, taxaNames);
                    if(taxonIndex == -1){
                        std::cerr << "Token '" + tok + "' is not in taxa names" << std::endl;
                        std::exit(1);
                    }
                    newNode->id = taxonIndex;

                    p = newNode;
                }
            }
            readingBranchLength = false;
        }
    }

    int id = tips.size();
    for(auto& n : nodes){
        if(n->isTip == false){
            n->id = id++;
        }
    }

    std::sort(nodes.begin(), nodes.end(),
        [](const std::unique_ptr<TreeNode>& a, const std::unique_ptr<TreeNode>& b) {
            return a->id < b->id;
        }
    );

    regeneratePostOrder();
}

TreeNode* Tree::addNode() {
    auto generator = std::mt19937(std::random_device{}());
    std::exponential_distribution<double> branchDist(10.0);

    auto newNode = std::make_unique<TreeNode>(
        TreeNode{0, "", false, false, branchDist(generator), nullptr, {}, false, false}
    );

    TreeNode* rawPtr = newNode.get();
    nodes.push_back(std::move(newNode));

    return rawPtr;
}

int Tree::getTaxonIndex(std::string token, std::vector<std::string> taxaNames){

    for(int i = 0, n = taxaNames.size(); i < n; i++){
        if(taxaNames[i] == token)
            return i;
    }

    return -1;
}

std::vector<std::string> Tree::parseNewickString(std::string newick){
    std::vector<std::string> tokens;
    std::string str = "";
    for(int i = 0; i < newick.length(); i++){
        char c = newick[i];
        if(c == '(' || c == ')' || c == ',' || c == ':' || c == ';'){
            if(str != ""){
                tokens.push_back(str);
                str = "";
            }

            tokens.push_back(std::string(1, c));
        }
        else {
            str += std::string(1, c);
        }
    }

    return tokens;
}

Tree::Tree(std::vector<std::string> taxaNames) : Tree(taxaNames.size()){
    for(int i = 0; i < tips.size(); i++){
        tips[i]->name = taxaNames[i];
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

std::string complementBinary(std::string s){
    std::string r = "";
    for(char c : s){
        if(c == '0')
            r += '1';
        else
            r += '0';
    }
    return r;
}

std::set<std::string> Tree::getSplits(){
    std::set<std::string> splits;
    std::unordered_map<TreeNode*, std::string> splitString;
    std::string zeroString = "";
    for(int i = 0; i < tips.size(); i++)
        zeroString += "0";

    for(int i = 0; i < tips.size(); i++){
        std::string tipString(zeroString);
        tipString[i] = '1';
        splitString[nodes[i].get()] = tipString;
    }

    for(TreeNode* n : postOrder){
        if(!n->isTip && !n->isRoot){
            std::string newString(zeroString);
            for(TreeNode* d : n->descendants){
                std::string descString = splitString[d];
                for(int i = 0; i < tips.size(); i++){
                    if(descString[i] == '1'){
                        newString[i] = '1';
                    }
                }
            }
            std::string complementS = complementBinary(newString);
            if(complementS < newString)
                splits.insert(complementS);
            else
                splits.insert(newString);
            
            splitString[n] = newString;
        }
    }

    return splits;
}

std::vector<std::string> Tree::getInternalSplitVec(){
    // We are only able to explicitly skip the root this way because of how we assigned IDs above
    std::vector<std::string> splits(nodes.size() - tips.size() - 1, "");
    std::unordered_map<TreeNode*, std::string> splitString;
    std::string zeroString = "";
    for(int i = 0; i < tips.size(); i++)
        zeroString += "0";

    for(int i = 0; i < tips.size(); i++){
        std::string tipString(zeroString);
        tipString[i] = '1';
        splitString[nodes[i].get()] = tipString;
    }

    for(TreeNode* n : postOrder){
        if(!n->isTip && !n->isRoot){
            std::string newString(zeroString);
            for(TreeNode* d : n->descendants){
                std::string descString = splitString[d];
                for(int i = 0; i < tips.size(); i++){
                    if(descString[i] == '1'){
                        newString[i] = '1';
                    }
                }
            }
            std::string complementS = complementBinary(newString);
            if(complementS < newString)
                splits[n->id - tips.size() - 1] = complementS;
            else
                splits[n->id - tips.size() - 1] = newString;
            
            splitString[n] = newString;
        }
    }

    return splits;
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

// Biased interior node selection using the oosterior probability of the split that it defines 
double Tree::adaptiveNNIMove(double epsilon, std::mt19937& gen, const std::unordered_map<std::string, double>& splitPosterior){
    std::uniform_real_distribution unifDist(0.0, 1.0);
    
    std::vector<std::string> currentSplits = getInternalSplitVec();
    std::vector<double> nodeProbs(currentSplits.size(), 0.0);

    double total = 0.0;
    for(TreeNode* n : postOrder){
        if(!n->isRoot && !n->isTip){
            int internalID = n->id - tips.size() - 1;
            total += epsilon;
            nodeProbs[internalID] = epsilon;
            if(splitPosterior.contains(currentSplits[internalID])){
                double posteriorContribution = 1.0 - splitPosterior.at(currentSplits[internalID]);
                nodeProbs[internalID] += posteriorContribution;
                total += posteriorContribution;
            }
            else {
                total += 1.0;
                nodeProbs[internalID] += 1.0;
            }
        }
    }

    double randomNode = unifDist(gen);
    double cumulativeSum = 0.0;
    double selectedProb = 0.0;
    int nodeID = 0;
    for(int i = 0; i < nodeProbs.size(); i++){
        double prob = nodeProbs[i]/total;
        cumulativeSum += prob;
        if(randomNode <= cumulativeSum){
            selectedProb = prob;
            nodeID = i + tips.size() + 1; // Remember that we are only considering the interior
            break;
        }
    }

    TreeNode* p = nodes[nodeID].get();
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

    // Refresh the current splits and probabilities to compute the hastings ratio
    currentSplits = getInternalSplitVec();
    total = 0.0;
    for(TreeNode* n : postOrder){
        if(!n->isRoot && !n->isTip){
            int internalID = n->id - tips.size() - 1;
            total += epsilon;
            nodeProbs[internalID] = epsilon;
            if(splitPosterior.contains(currentSplits[internalID])){
                double posteriorContribution = 1.0 - splitPosterior.at(currentSplits[internalID]);
                nodeProbs[internalID] += posteriorContribution;
                total += posteriorContribution;
            }
            else {
                total += 1.0;
                nodeProbs[internalID] += 1.0;
            }
        }
    }
    double revProb = nodeProbs[nodeID - tips.size() - 1] / total;

    return std::log(revProb) - std::log(selectedProb);
}