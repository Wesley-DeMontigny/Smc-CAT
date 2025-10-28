#ifndef TREE_HPP
#define TREE_HPP
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <string>
#include <set>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * @brief 
 * 
 */
struct TreeNode {
    int id;
    std::string name;
    bool isTip;
    bool isRoot;
    double branchLength;

    // With these raw pointers we must require that our tree node's access will not outlive the node vector
    TreeNode* ancestor;
    std::set<TreeNode*> descendants;

    bool updateCL;
    bool updateTP;
};

struct NodeHash {
    size_t operator()(const TreeNode* n) const noexcept {
        return std::hash<int>{}(n->id);
    }
};

/**
 * @brief 
 * 
 */
class Tree {
    public:
        Tree(void)=delete;
        Tree(boost::random::mt19937& rng, int n); // Generate random tree with n tips
        Tree(std::string s, const std::vector<std::string>& taxaNames); // Generate a tree from newick
        Tree(boost::random::mt19937& rng, const std::vector<std::string>& taxaNames); // Generate a tree from a list of taxa
        Tree(const std::vector<boost::dynamic_bitset<>>& splits, const std::vector<std::string>& taxaNames);
        Tree(const Tree& t); // Deep copy
        Tree& operator=(const Tree& t); // Assignment copy

        void assignMeanBranchLengths(const std::vector<std::string>& newickStrings, const std::vector<double>& normalizedWeights, const std::vector<std::string>& taxaNames);
        std::string generateNewick(const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities) const;
        std::string generateNewick() const; // Generates a Newick string for the tree. Assumes the post-order is correct.
        const std::vector<TreeNode*>& getPostOrder() const {return postOrder;}
        const std::vector<TreeNode*>& getTips() const {return tips;}
        std::set<boost::dynamic_bitset<>> getSplits() const;
        std::vector<boost::dynamic_bitset<>> getSplitVector() const;

        const int getNumNodes() const {return postOrder.size();}
        const int getNumTaxa() const {return tips.size();}
        TreeNode* getRoot() const {return root;}

        double scaleBranchMove(boost::random::mt19937& rng, double delta);
        double scaleSubtreeMove(boost::random::mt19937& rng, double delta);
        double adaptiveNNIMove(boost::random::mt19937& rng, double epsilon, const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosterior);
        double NNIMove(boost::random::mt19937& rng);

        void updateAll();
    private:
        void regeneratePostOrder();
        void clone(const Tree& t);

        TreeNode* root;
        std::vector<TreeNode*> postOrder;
        std::vector<TreeNode*> tips;
        std::vector<std::unique_ptr<TreeNode>> nodes; // We let the nodes vector own the node unique_ptr
        
        TreeNode* addNode(boost::random::mt19937& rng);
        TreeNode* addNode();
        TreeNode* buildTree(const std::vector<boost::dynamic_bitset<>>& splits, const boost::dynamic_bitset<>& taxa);
        int getTaxonIndex(std::string token, const std::vector<std::string>& taxaNames) const;
        std::vector<std::string> parseNewickString(const std::string newick) const;

        void recursivePostOrderAssign(TreeNode* p);
boost::dynamic_bitset<> parseAndAccumulate(const std::vector<std::string>& tokens,
                         double normalizedWeight,
                         const std::vector<std::string>& taxaNames,
                         std::unordered_map<boost::dynamic_bitset<>, boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::weighted_mean>, double>>& branchMeans);
        std::string recursiveNewickGenerate(std::string s, TreeNode* p) const;
        std::string recursiveNewickGenerate(std::string s, TreeNode* p, const std::unordered_map<boost::dynamic_bitset<>, double>& splitPosteriorProbabilities, const std::vector<boost::dynamic_bitset<>>& splitVec) const;
};

#endif