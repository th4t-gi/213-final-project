#include "utils.h"

/**
 * Generates branches of phylogenetic tree from prufer sequence
 *
 * \param N number of leaves in tree
 * \param seq prufer sequence to convert
 * \param branches array to save branch data to
 * \param degrees array to save degree data to
 */
void prufer_seq_to_tree(const int N, prufer_t seq, code_t branches[], degree_t degrees[]) {
  int prufer_max = *std::max_element(seq.begin(), seq.end());

  // add implied edge to prufer sequence
  seq.push_back(1);

  for (size_t i = 0; i < seq.size(); i++) {
    uint8_t branch_index = seq[i] - 1;
    // for each leave index, add the binary encoded value to the branch index
    if (i < N) {
      branches[branch_index] += (1 << i);
    } else {
      // for indecies in the prufer seq that refer to branches, find index of parent and add its sum to parent
      branches[branch_index] += branches[prufer_max + N - i - 1];
    }
  }

  // count appearances of each index for the degrees of each branch
  for (int i = 0; i < prufer_max; i++) {
    degree_t d = std::count(seq.begin(), seq.end(), i+1);
    degrees[i] = d;
  }
}