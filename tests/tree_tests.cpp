#include <gtest/gtest.h>
#include "analysis/Tree.hpp"

class TreeTest : public ::testing::Test {
protected:
    void SetUp() override {

    }

    Tree randomTree{25};
};

// Make sure that every node except for the root has only two children
TEST_F(TreeTest, RandomTreeBifurcating) {
    bool bifurcating = true;
    auto postOrder = randomTree.getPostOrder();
    for(auto node : postOrder){
        if(! node->isRoot && ! node->isTip){
            if(node->descendants.size() != 2){
                bifurcating = false;
                break;
            }
        }
    }

    EXPECT_EQ(bifurcating, true);
}

// Make sure we have initialized the right number of tips
TEST_F(TreeTest, RandomTreeTipCount) {
    int size = randomTree.getTips().size();
    EXPECT_EQ(size, 25);
}