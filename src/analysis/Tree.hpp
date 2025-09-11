#ifndef TREE_HPP
#define TREE_HPP
#include <string>
#include <set>
#include <vector>
#include <memory>

/*

*/
struct TreeNode {
    int id;
    std::string name;
    bool isTip;
    bool isRoot;
    double branchLength;

    std::shared_ptr<TreeNode> ancestor;
    std::set<std::shared_ptr<TreeNode>> descendants;

    bool updateCL;
    bool updateTP;
};

/*

*/
class Tree {
    public:
        Tree(void)=delete;
        Tree(int n); // Generate random tree with n tips
        Tree(std::string s); // Generate a tree from newick
        Tree(std::vector<std::string> tList); // Generate a tree from a list of taxa
        Tree(const Tree& t); // Deep copy
        Tree& operator=(const Tree& t);
        ~Tree();

        std::string generateNewick(); // Generates a Newick string for the tree. Assumes the post-order is correct.
        std::vector<std::shared_ptr<TreeNode>>& getPostOrder() {return postOrder;}
        std::vector<std::shared_ptr<TreeNode>>& getTips() {return tips;}

        int getNumNodes() {return postOrder.size();}
        int getNumTaxa() {return tips.size();}

        double localMove(double delta);
        double scaleBranchMove(double delta);
        double NNI();

        void updateAll();
    private:
        void regeneratePostOrder();
        void clone(const Tree& t);

        std::shared_ptr<TreeNode> root;
        std::vector<std::shared_ptr<TreeNode>> postOrder;
        std::vector<std::shared_ptr<TreeNode>> tips;
        std::vector<std::shared_ptr<TreeNode>> nodes;

        void recursivePostOrderAssign(std::shared_ptr<TreeNode> p);
        int recursiveIDAssign(int n, std::shared_ptr<TreeNode> p);
        std::string recursiveNewickGenerate(std::string s, std::shared_ptr<TreeNode> p);
};

#endif