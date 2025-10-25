#ifndef TREE_HPP
#define TREE_HPP
#include <boost/random/mersenne_twister.hpp>
#include <string>
#include <set>
#include <vector>
#include <memory>
#include <unordered_map>

/*

*/
struct TreeNode {
    int id;
    std::string name;
    bool isTip;
    bool isRoot;
    double branchLength;

    // With these raw pointers we are enforcing that our tree node's access will not outlive the node vector
    TreeNode* ancestor;
    std::set<TreeNode*> descendants;

    bool updateCL;
    bool updateTP;
};

/*

*/
class Tree {
    public:
        Tree(void)=delete;
        Tree(boost::random::mt19937& rng, int n); // Generate random tree with n tips
        Tree(std::string s, std::vector<std::string> taxaNames); // Generate a tree from newick
        Tree(boost::random::mt19937& rng, std::vector<std::string> taxaNames); // Generate a tree from a list of taxa
        Tree(const Tree& t); // Deep copy
        Tree& operator=(const Tree& t);
        ~Tree();

        std::string generateNewick(std::unordered_map<std::string, double>& splitPosteriorProbabilities);
        std::string generateNewick(); // Generates a Newick string for the tree. Assumes the post-order is correct.
        std::vector<TreeNode*>& getPostOrder() {return postOrder;}
        std::vector<TreeNode*>& getTips() {return tips;}
        std::set<std::string> getSplits();
        std::vector<std::string> getInternalSplitVec();

        const int getNumNodes() {return postOrder.size();}
        const int getNumTaxa() {return tips.size();}
        TreeNode* getRoot() {return root;}

        double scaleBranchMove(boost::random::mt19937& rng, double delta);
        double scaleSubtreeMove(boost::random::mt19937& rng, double delta);
        double adaptiveNNIMove(boost::random::mt19937& rng, double epsilon,const std::unordered_map<std::string, double>& splitPosterior);
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
        int getTaxonIndex(std::string token, std::vector<std::string> taxaNames);
        std::vector<std::string> parseNewickString(std::string newick);

        void recursivePostOrderAssign(TreeNode* p);
        std::string recursiveNewickGenerate(std::string s, TreeNode* p);
        std::string recursiveNewickGenerate(std::string s, TreeNode* p, std::unordered_map<std::string, double>& splitPosteriorProbabilities, std::vector<std::string>& splitVec);
};

#endif