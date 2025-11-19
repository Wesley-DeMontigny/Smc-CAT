#include "Tree.hpp"
#include <algorithm>
#include <boost/random/uniform_01.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <cassert>
#include <iostream>
#include <memory>

// Construct random tree with n tips and exponentially distributed branch lengths
Tree::Tree(boost::random::mt19937& rng, int n){
    assert(n >= 3);

    boost::random::uniform_01<double> unif{};

    root = addNode(rng);
    root->isRoot = true;
    root->branchLength = 0.0;
    auto A = addNode(rng);
    A->isTip = true;
    A->ancestor = root;
    A->name = "t0";
    A->id = 0;
    auto B = addNode(rng);
    B->isTip = true;
    B->ancestor = root;
    B->name = "t1";
    B->id = 1;
    
    root->descendants.insert(A);
    root->descendants.insert(B);

    tips.push_back(A);
    tips.push_back(B);

    for(int i = 2; i < n; i++){
        int randomIndex = static_cast<int>(unif(rng) * i);
        auto randomTip = tips[randomIndex];

        auto newInternal = addNode(rng);
        newInternal->ancestor = randomTip->ancestor;
        auto newTip = addNode(rng);
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

Tree::Tree(std::string newick, const std::vector<std::string>& taxaNames){
    std::vector<std::string> tokens = parseNewickString(newick);

    TreeNode* p = nullptr;
    bool readingBranchLength = false;

    for(std::string tok : tokens){
        if(tok == "("){
            TreeNode* newNode = addNode();
            if(p == nullptr){
                root = newNode;
                newNode->branchLength = 0.0;
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

Tree::Tree(boost::random::mt19937& rng, const std::vector<std::string>& taxaNames) : Tree(rng, taxaNames.size()){
    for(int i = 0; i < tips.size(); i++){
        tips[i]->name = taxaNames[i];
    }
}

Tree::Tree(const std::vector<std::pair<boost::dynamic_bitset<>, double>>& splits, const std::vector<std::string>& taxaNames) {

    for(int i = 0; i < taxaNames.size(); i++){
        auto newTip = addNode();
        newTip->isTip = true;
        newTip->name = taxaNames[i];
        newTip->id = i;
        tips.push_back(newTip);
    }

    boost::dynamic_bitset<> all(taxaNames.size());
    all.flip();
    root = buildTree(splits, all);
    root->isRoot = true;
    root->branchLength = 0.0;

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

Tree::Tree(const Tree& t) {
    root = addNode();
    root->isRoot = true;
    root->branchLength = 0.0;

    auto initTip0 = addNode();
    initTip0->id = 0;
    initTip0->isTip = true;
    auto initTip1 = addNode();
    initTip1->id = 1;
    initTip1->isTip = true;

    tips.push_back(initTip0);
    tips.push_back(initTip1);

    for(int i = 2; i < t.tips.size(); i++){
        addNode();
        auto newTip = addNode();
        newTip->id = i;
        newTip->isTip = true;
        tips.push_back(newTip);
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

TreeNode* Tree::buildTree(const std::vector<std::pair<boost::dynamic_bitset<>, double>>& splits, const boost::dynamic_bitset<>& taxa){
    size_t taxaSize = taxa.count();
    if(taxaSize > 1){
        auto newNode = addNode();

        std::vector<std::pair<boost::dynamic_bitset<>, double>> relevant;
        relevant.reserve(splits.size());
        for (const auto& s : splits) {
            if ((s.first & taxa).any() && ((~s.first) & taxa).any())
                relevant.push_back(s);
        }

        auto& s = relevant.front();
        boost::dynamic_bitset<> left  = s.first & taxa;
        boost::dynamic_bitset<> right = (~s.first) & taxa;

        auto leftNode = buildTree(relevant, left);
        auto rightNode = buildTree(relevant, right);
        leftNode->ancestor = newNode;
        rightNode->ancestor = newNode;
        newNode->descendants.insert(leftNode);
        newNode->descendants.insert(rightNode);
        if(taxa.all()){ // We are at the root and need to split the branch length (the Pulley Principle)
            leftNode->branchLength = s.second / 2.0;
            rightNode->branchLength = s.second / 2.0;
        }
        else{
            bool setRight = false;
            bool setLeft = false;
            boost::dynamic_bitset<> rightSearch{right};
            boost::dynamic_bitset<> leftSearch{left};

            // Canonicalize bitset
            if(!rightSearch.test(0))
                rightSearch.flip();
            if(!leftSearch.test(0))
                leftSearch.flip();

            for(const auto& [splitBits, length] : relevant) {
                if(!setLeft && splitBits == leftSearch) {
                    leftNode->branchLength = length;
                    setLeft = true;
                }
                if(!setRight && splitBits == rightSearch) {
                    rightNode->branchLength = length;
                    setRight = true;
                }
                if(setLeft && setRight) break;
            }
        }

        return newNode;
    }
    else{
        return tips[taxa.find_first()];
    }
}

TreeNode* Tree::addNode(boost::random::mt19937& rng) {

    auto newNode = std::make_unique<TreeNode>(
        TreeNode{0, "", false, false, boost::random::exponential_distribution<double>{10.0}(rng), nullptr, {}, false, false}
    );

    TreeNode* rawPtr = newNode.get();
    nodes.push_back(std::move(newNode));

    return rawPtr;
}

TreeNode* Tree::addNode() {

    auto newNode = std::make_unique<TreeNode>(
        TreeNode{0, "", false, false, 1.0, nullptr, {}, false, false}
    );

    TreeNode* rawPtr = newNode.get();
    nodes.push_back(std::move(newNode));

    return rawPtr;
}

int Tree::getTaxonIndex(std::string token, const std::vector<std::string>& taxaNames) const{

    for(int i = 0, n = taxaNames.size(); i < n; i++){
        if(taxaNames[i] == token)
            return i;
    }

    return -1;
}

std::vector<std::string> Tree::parseNewickString(const std::string newick) const {
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

std::string Tree::generateNewick() const {
    std::string output = "";

    output = recursiveNewickGenerate(output, root);

    output += ";";

    return output;
}

std::string Tree::generateNewick(const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities) const{
    std::string output = "";

    output = recursiveNewickGenerate(output, root, splitPosteriorProbabilities, computeSplits());

    output += ";";

    return output;
}

std::string Tree::recursiveNewickGenerate(std::string s, TreeNode* p) const{
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

std::string Tree::recursiveNewickGenerate(std::string s, TreeNode* p, const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities, const std::unordered_map<int, boost::dynamic_bitset<>>& splitMap) const{
    if(! p->isTip){
        s += "(";
        for(auto child : p->descendants) {
            s = recursiveNewickGenerate(s, child, splitPosteriorProbabilities, splitMap);
            s += ",";
        }
        s.pop_back();
        s += ")";

        if(p != root){
            auto& split = splitMap.at(p->id);
            s += std::to_string(splitPosteriorProbabilities.at(split));
            s += ":" + std::to_string(p->branchLength);
        }
    }
    else {
        s += p->name + ":" + std::to_string(p->branchLength);
    }

    return s;
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

void Tree::updateAll(){
    for(auto n : postOrder){
        n->updateCL = true;
        n->updateTP = true;
    }
}

std::unordered_map<int, boost::dynamic_bitset<>> Tree::computeSplits() const {
    std::unordered_map<int, boost::dynamic_bitset<>> splits;
    boost::dynamic_bitset<> zeroSet = boost::dynamic_bitset<>{tips.size()}; 

    for(int i = 0; i < tips.size(); i++){ 
        boost::dynamic_bitset<> tipSet(zeroSet); 
        tipSet[i] = 1; 
        splits.emplace(nodes[i]->id, tipSet);
    } 


    for(TreeNode* n : postOrder){ 
        if(!n->isRoot){ 
            boost::dynamic_bitset<> newSet(zeroSet); 
            for(TreeNode* d : n->descendants){ 
                boost::dynamic_bitset<> descSet = splits[d->id]; 
                for(int i = 0; i < tips.size(); i++){ 
                    if(descSet.test(i)){ 
                        newSet[i] = 1; 
                    } 
                } 
            } 
            splits.emplace(n->id, newSet); 
        } 
    } 
    
    // Canonicalize bitset 
    for(auto& [_, s] : splits){ 
        if(!s.test(0)){ 
            s.flip(); 
        } 
    }

    return splits;
}

std::unordered_map<int, boost::dynamic_bitset<>> Tree::getMoveableSplits() const {
    auto splits = computeSplits();
    for(auto it = splits.begin(); it != splits.end();){
        const auto& s = it->second;
        if(s.count() == 1 || (~s).count() == 1)
            it = splits.erase(it);
        else
            ++it;
    }
    return splits;
}

std::set<boost::dynamic_bitset<>> Tree::getSplitSet() const {
    auto splits = computeSplits();
    std::set<boost::dynamic_bitset<>> cladeSet;
    for(auto& [_, s] : splits)
        cladeSet.insert(s);
    return cladeSet;
}

std::unordered_map<boost::dynamic_bitset<>, double> Tree::getSplitBranchMap() const {
    auto splits = computeSplits();
    std::unordered_map<boost::dynamic_bitset<>, double> branchMap;

    for(const auto& [id, s] : splits){
        double len = nodes[id]->branchLength;
        branchMap[s] += len;
    }

    return branchMap;
}

double Tree::scaleBranchMove(boost::random::mt19937& rng, double delta){
    boost::random::uniform_01<double> unif{};


    TreeNode* p = nullptr;
    do{
        p = nodes[static_cast<int>(unif(rng) * nodes.size())].get();
    }
    while(p == root);

    double currentV = p->branchLength;
    double scale = std::exp(delta * (unif(rng) - 0.5));
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

double Tree::scaleSubtreeMove(boost::random::mt19937& rng, double delta){
    boost::random::uniform_01<double> unif{};

    TreeNode* p = nullptr;
    do{
        p = nodes[static_cast<int>(unif(rng) * nodes.size())].get();
    }
    while(p == root);

    double scale = std::exp(delta * (unif(rng) - 0.5));
    int numBranches = scaleSubtreeRecurse(p, scale);

    TreeNode* q = p->ancestor;
    while(q != root){
        q->updateCL = true;
        q = q->ancestor;
    }
    root->updateCL = true;

    return numBranches * std::log(scale);
}

TreeNode* chooseNodeFromSet(boost::random::mt19937& rng, std::set<TreeNode*>& s){
    boost::random::uniform_01<double> unif{};

    int index = static_cast<int>(unif(rng) * s.size());

    auto it = s.begin();
    std::advance(it, index);

    return *it;
}

double Tree::NNIMove(boost::random::mt19937& rng){
    boost::random::uniform_01<double> unif{};


    TreeNode* p = nullptr;
    do{
        p = nodes[static_cast<int>(unif(rng) * nodes.size())].get();
    }
    while(p == root || p->isTip);

    TreeNode* a = p->ancestor;

    std::set<TreeNode*> neighbors1 = p->descendants;
    TreeNode* n1 = chooseNodeFromSet(rng, neighbors1);

    // We exclude
    std::set<TreeNode*> neighbors2 = a->descendants;
    neighbors2.erase(p);
    TreeNode* n2 = chooseNodeFromSet(rng, neighbors2);

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

// Biased interior node selection using the posterior probability of the split that node creates. Inspired by X Meyer 2021
double Tree::adaptiveNNIMove(boost::random::mt19937& rng, double epsilon, const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosterior){
    boost::random::uniform_01<double> unif{};
    
    std::unordered_map<int, boost::dynamic_bitset<>> currentSplits = getMoveableSplits();
    std::unordered_map<int, double> nodeProbs;

    double total = 0.0;
    for(TreeNode* n : postOrder){
        if(!n->isRoot && !n->isTip){
            total += epsilon;
            nodeProbs[n->id] = epsilon;
            if(splitPosterior.contains(currentSplits[n->id])){
                double posteriorContribution = 1.0 - splitPosterior.at(currentSplits[n->id]);
                nodeProbs[n->id] += posteriorContribution;
                total += posteriorContribution;
            }
            else {
                total += 1.0;
                nodeProbs[n->id] += 1.0;
            }
        }
    }

    double randomNode = unif(rng);
    double cumulativeSum = 0.0;
    double selectedProb = 0.0;
    int nodeID = 0;
    for(const auto& [index, probs] : nodeProbs){
        double prob = nodeProbs[index]/total;
        cumulativeSum += prob;
        if(randomNode <= cumulativeSum){
            selectedProb = prob;
            nodeID = index;
            break;
        }
    }

    TreeNode* p = nodes[nodeID].get();
    TreeNode* a = p->ancestor;

    std::set<TreeNode*> neighbors1 = p->descendants;
    TreeNode* n1 = chooseNodeFromSet(rng, neighbors1);

    // We exclude
    std::set<TreeNode*> neighbors2 = a->descendants;
    neighbors2.erase(p);
    TreeNode* n2 = chooseNodeFromSet(rng, neighbors2);

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
    currentSplits = getMoveableSplits();
    total = 0.0;
    for(TreeNode* n : postOrder){
        if(!n->isRoot && !n->isTip){
            total += epsilon;
            nodeProbs[n->id] = epsilon;
            if(splitPosterior.contains(currentSplits[n->id])){
                double posteriorContribution = 1.0 - splitPosterior.at(currentSplits[n->id]);
                nodeProbs[n->id] += posteriorContribution;
                total += posteriorContribution;
            }
            else {
                total += 1.0;
                nodeProbs[n->id] += 1.0;
            }
        }
    }
    double revProb = nodeProbs[nodeID] / total;

    return std::log(revProb) - std::log(selectedProb);
}

double Tree::SPRMove(boost::random::mt19937& rng) {
    boost::random::uniform_01<double> unif{};

    TreeNode* p = nullptr;
    do {
        p = nodes[static_cast<int>(unif(rng) * nodes.size())].get();
    } while (p == root || p->isTip);

    std::set<TreeNode*> children = p->descendants;
    TreeNode* s = chooseNodeFromSet(rng, children);

    TreeNode* sibling = nullptr; // Here we are assuming a strictly binary tree
    for (auto* c : p->descendants) {
        if (c != s) sibling = c;
    }

    TreeNode* a = nullptr; // Select attachment point
    do {
        a = nodes[static_cast<int>(unif(rng) * nodes.size())].get();
    } while (a->isInSubtree(p) || a == root);
    TreeNode* aParent = a->ancestor;

    p->descendants.erase(sibling); // Pick up subtree by p
    sibling->ancestor = p->ancestor;
    p->ancestor->descendants.insert(sibling);
    p->ancestor->descendants.erase(p);
    p->ancestor = nullptr;

    // Attach the subtree
    p->ancestor = aParent;
    aParent->descendants.erase(a);
    aParent->descendants.insert(p);
    p->descendants.insert(a);
    a->ancestor = p;

    TreeNode* q = p;
    while (q != nullptr) {
        q->updateCL = true;
        q = q->ancestor;
    }

    regeneratePostOrder();

    return 0.0;
}