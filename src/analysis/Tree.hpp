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
};

/*

*/
class Tree {
    public:
        Tree(void)=delete;
        Tree(int n); // Generate random tree with n tips
        Tree(std::string s); // Generate a tree from newick
        ~Tree();

        std::string generateNewick(); // Generates a Newick string for the tree. Assumes the post-order is correct.
        std::vector<std::shared_ptr<TreeNode>>& getPostOrder() {return postOrder;}
        std::vector<std::shared_ptr<TreeNode>>& getTips() {return tips;}
    private:
        void regeneratePostOrder();

        std::shared_ptr<TreeNode> root;
        std::vector<std::shared_ptr<TreeNode>> postOrder;
        std::vector<std::shared_ptr<TreeNode>> tips;

        void recursivePostOrderAssign(std::shared_ptr<TreeNode> p);
        int recursiveIDAssign(int n, std::shared_ptr<TreeNode> p);
        std::string recursiveNewickGenerate(std::string s, std::shared_ptr<TreeNode> p);
};

#endif