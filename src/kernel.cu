#include "kernel.h"
#include <algorithm>


__device__ tree_map_t prufer_seq_to_branches(int N, prufer_t seq) {
    int prufer_max = *std::max_element(seq.begin(), seq.end());

    // uint8_t max = 1;
    seq.push_back(1);
    
    //https://stackoverflow.com/questions/13110130/initialize-a-vector-to-zeros-c-c11
    std::vector<code_t> branches(prufer_max, 0);

    for (int i = 0; i < seq.size(); i++) {
        uint8_t branch_index = seq[i] - 1;
        // for each leave index, add the binary encoded value to the branch index
        if (i < N) {
            branches[branch_index] += 2 << i;
        } else {
            // for indecies in the prufer seq that refer to branches, find index of parent and add its sum to parent
            branches[branch_index] += branches[prufer_max + N - i - 1];
        }
    }

    tree_map_t tree;
    tree.reserve(prufer_max);
    // count appearances of each index for the degrees of each branch
    for (int i = 0; i < branches.size(); i++) {
        int J = branches[i];
        int degrees = std::count(seq.begin(), seq.end(), i+1);
        tree.insert(J, degrees);
    }

    return tree;
}