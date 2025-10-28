#include "Tree.hpp"
#include <algorithm>
#include <boost/accumulators/statistics/sum.hpp>
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

Tree::Tree(const std::vector<boost::dynamic_bitset<>>& splits, const std::vector<std::string>& taxaNames) {

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
        auto newInternal = addNode();
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

void Tree::assignMeanBranchLengths(const std::vector<std::string>& newickStrings, const std::vector<double>& normalizedWeights, const std::vector<std::string>& taxaNames){
    using Acc = boost::accumulators::accumulator_set<
        double, 
        boost::accumulators::stats<boost::accumulators::tag::weighted_mean>, 
        double
    >;

    std::unordered_map<boost::dynamic_bitset<>, Acc> branchMeans{};
    std::unordered_map<TreeNode*, boost::dynamic_bitset<>, NodeHash> splitMap;
    boost::dynamic_bitset<> zeroSet = boost::dynamic_bitset<>{tips.size()};

    for(int i = 0; i < tips.size(); i++){
        boost::dynamic_bitset<> tipSet(zeroSet);
        tipSet[i] = 1;
        splitMap[nodes[i].get()] = tipSet;
    }

    for(TreeNode* n : postOrder){
        if(!n->isTip && !n->isRoot){
            boost::dynamic_bitset<> newSet(zeroSet);
            for(TreeNode* d : n->descendants){
                boost::dynamic_bitset<> descSet = splitMap[d];
                for(int i = 0; i < tips.size(); i++){
                    if(descSet[i] == 1){
                        newSet[i] = 1;
                    }
                }
            }
            splitMap[n] = newSet;
        }
    }

    for(const auto& s : splitMap) {
        branchMeans.emplace(s.second, Acc{});
    }

    for(int i = 0; i < newickStrings.size(); i++){
        std::vector<std::string> tokens = parseNewickString(newickStrings[i]);
        parseAndAccumulate(tokens, normalizedWeights[i], taxaNames, branchMeans);
    }

    for(auto& pair : splitMap){
        pair.first->branchLength = boost::accumulators::weighted_mean(branchMeans[pair.second]);
    }
}

TreeNode* Tree::buildTree(const std::vector<boost::dynamic_bitset<>>& splits, const boost::dynamic_bitset<>& taxa){
    size_t taxaSize = taxa.count();
    if(taxaSize > 2){
        auto newNode = addNode();

        std::vector<boost::dynamic_bitset<>> relevant;
        relevant.reserve(splits.size());
        for (const auto& s : splits) {
            if ((s & taxa).any() && ((~s) & taxa).any())
                relevant.push_back(s);
        }

        auto& s = relevant.front();
        boost::dynamic_bitset<> left  = s & taxa;
        boost::dynamic_bitset<> right = (~s) & taxa;

        auto leftNode = buildTree(relevant, left);
        auto rightNode = buildTree(relevant, right);
        leftNode->ancestor = newNode;
        rightNode->ancestor = newNode;
        newNode->descendants.insert(leftNode);
        newNode->descendants.insert(rightNode);

        return newNode;
    }
    else if(taxaSize == 2){
        auto newNode = addNode();

        for(int i = 0; i < taxa.size(); i++){
            if(taxa[i] == 1){
                tips[i]->ancestor = newNode;
                newNode->descendants.insert(tips[i]);
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
    std::vector<boost::dynamic_bitset<>> splits = getSplitVector();

    std::string output = "";

    output = recursiveNewickGenerate(output, root, splitPosteriorProbabilities, splits);

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

void Tree::parseAndAccumulate(const std::vector<std::string>& tokens, double normalizedWeight, const std::vector<std::string>& taxaNames, std::unordered_map<boost::dynamic_bitset<>, boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::weighted_mean>, double>>& branchMeans) const {
    int numTaxa = taxaNames.size();
    std::vector<boost::dynamic_bitset<>> stack; // Holds splits as we go down the tree
    boost::dynamic_bitset<> emptySet{0};

    boost::dynamic_bitset<> lastCompletedSubtree;
    bool haveLast = false;

    bool readingBranchLength = false;
    double currentBranchLength = 0.0;

    for (int i = 0; i < tokens.size(); i++){
        const std::string& tok = tokens[i];

        if (tok == "("){
            stack.push_back(emptySet);
        }
        else if (tok == ")"){
            boost::dynamic_bitset<> clade(numTaxa);

            while (!stack.empty() && stack.back().size() != 0){
                clade |= stack.back();
                stack.pop_back();
            
            }
            if (stack.empty()) throw std::runtime_error("Unbalanced tree!");

            stack.pop_back();

            lastCompletedSubtree = clade;
            haveLast = true;
            stack.push_back(clade);
        }
        else if (tok == ","){
            continue;
        }
        else if (tok == ":"){
            readingBranchLength = true;
        }
        else if (tok == ";"){
            break;
        }
        else{
            if (readingBranchLength){
                currentBranchLength= std::atof(tok.c_str());
                if (haveLast){
                    auto key = lastCompletedSubtree;
                    auto complement = ~key;

                    if(branchMeans.count(key) > 0){
                        branchMeans[key](currentBranchLength, boost::accumulators::weight = normalizedWeight);
                    }
                    else if(branchMeans.count(complement) > 0){
                        branchMeans[complement](currentBranchLength, boost::accumulators::weight = normalizedWeight);
                    }
                }
                readingBranchLength = false;
            } 
            else{
                int tipIndex = getTaxonIndex(tok, taxaNames);
                boost::dynamic_bitset<> leaf(numTaxa);
                leaf.set(tipIndex);
                stack.push_back(std::move(leaf));
                lastCompletedSubtree = stack.back();
                haveLast = true;
            }
        }
    }
}

std::string Tree::recursiveNewickGenerate(std::string s, TreeNode* p, const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities, const std::vector<boost::dynamic_bitset<>>& splitVec) const{
    if(! p->isTip){
        s += "(";
        for(auto child : p->descendants) {
            s = recursiveNewickGenerate(s, child, splitPosteriorProbabilities, splitVec);
            s += ",";
        }
        s.pop_back();
        s += ")";

        if(p != root){
            s += std::to_string(splitPosteriorProbabilities.at(splitVec.at(p->id - tips.size() - 1)));
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

std::set<boost::dynamic_bitset<>> Tree::getSplits() const{
    std::set<boost::dynamic_bitset<>> splits;
    std::unordered_map<TreeNode*, boost::dynamic_bitset<>, NodeHash> splitMap;
    boost::dynamic_bitset<> zeroSet = boost::dynamic_bitset<>{tips.size()};

    for(int i = 0; i < tips.size(); i++){
        boost::dynamic_bitset<> tipSet(zeroSet);
        tipSet[i] = 1;
        splitMap[nodes[i].get()] = tipSet;
    }

    for(TreeNode* n : postOrder){
        if(!n->isTip && !n->isRoot){
            boost::dynamic_bitset<> newSet(zeroSet);
            for(TreeNode* d : n->descendants){
                boost::dynamic_bitset<> descSet = splitMap[d];
                for(int i = 0; i < tips.size(); i++){
                    if(descSet[i] == 1){
                        newSet[i] = 1;
                    }
                }
            }

            boost::dynamic_bitset<> complement(newSet);
            complement.flip();
            if(complement < newSet)
                splits.insert(complement);
            else
                splits.insert(newSet);
            
            splitMap[n] = newSet;
        }
    }

    return splits;
}

std::vector<boost::dynamic_bitset<>> Tree::getSplitVector() const {
    std::vector<boost::dynamic_bitset<>> splits(nodes.size() - tips.size() - 1, boost::dynamic_bitset{tips.size()});
    std::unordered_map<TreeNode*, boost::dynamic_bitset<>, NodeHash> splitMap;
    boost::dynamic_bitset<> zeroSet = boost::dynamic_bitset<>{tips.size()};

    for(int i = 0; i < tips.size(); i++){
        boost::dynamic_bitset<> tipSet(zeroSet);
        tipSet[i] = 1;
        splitMap[nodes[i].get()] = tipSet;
    }

    for(TreeNode* n : postOrder){
        if(!n->isTip && !n->isRoot){
            boost::dynamic_bitset<> newSet(zeroSet);
            for(TreeNode* d : n->descendants){
                boost::dynamic_bitset<> descSet = splitMap[d];
                for(int i = 0; i < tips.size(); i++){
                    if(descSet[i] == 1){
                        newSet[i] = 1;
                    }
                }
            }

            boost::dynamic_bitset complement(newSet);
            complement.flip();
            if(complement < newSet)
                splits[n->id - tips.size() - 1] = complement;
            else
                splits[n->id - tips.size() - 1] = newSet;

            splitMap[n] = newSet;
        }
    }

    return splits;
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

    
    std::vector<boost::dynamic_bitset<>> currentSplits = getSplitVector();
    std::vector<double> nodeProbs(currentSplits.size(), 0.0);

    double total = 0.0;
    for(TreeNode* n : postOrder){
        if(!n->isRoot && !n->isTip){
            int internalID = n->id - tips.size() - 1;
            total += epsilon;
            nodeProbs[internalID] = epsilon;
            if(splitPosterior.contains(currentSplits[internalID])){
                double posteriorContribution = std::clamp(1.0 - splitPosterior.at(currentSplits[internalID]), 0.0, 1.0);
                nodeProbs[internalID] += posteriorContribution;
                total += posteriorContribution;
            }
            else {
                total += 1.0;
                nodeProbs[internalID] += 1.0;
            }
        }
    }

    double randomNode = unif(rng);
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
    currentSplits = getSplitVector();
    total = 0.0;
    for(TreeNode* n : postOrder){
        if(!n->isRoot && !n->isTip){
            int internalID = n->id - tips.size() - 1;
            total += epsilon;
            nodeProbs[internalID] = epsilon;
            if(splitPosterior.contains(currentSplits[internalID])){
                double posteriorContribution = std::clamp(1.0 - splitPosterior.at(currentSplits[internalID]), 0.0, 1.0);
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