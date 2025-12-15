#include <array>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

#include "utils.h"

#define GPU_BATCH_SIZE 100
#define MAX_THREADS 8

//https://www.geeksforgeeks.org/cpp/std-mutex-in-cpp/
std::mutex perm_mtx;
long permutations = 0;


// counted by https://oeis.org/A000070
prufer_arr_t generate_Lk(int k) {
  // BASE CASE:
  if (k <= 2) {
    prufer_arr_t base{{1}};
    return base;
  }

  // RECURSIVE CASE:
  prufer_arr_t Lk = {};

  for (prufer_t S : generate_Lk(k - 1)) {
    S.insert(S.begin(), 1);
    Lk.push_back(S);

    int ones = std::count(S.begin(), S.end(), 1);
    int twos = std::count(S.begin(), S.end(), 2);

    if (ones < 2 || ones >= twos) {
      prufer_t S2 = S;
      for (int i = 0; i < S2.size(); i++) {
        S2[i]++;
      }
      S2.insert(S2.begin(), 1);
      Lk.push_back(S2);
    }
  }

  return Lk;
}


prufer_arr_t generate_Lk_with_dup(int k) {
  // BASE CASE:
  if (k <= 2) {
    prufer_arr_t base{{1}};
    return base;
  }

  // RECURSIVE CASE:
  prufer_arr_t Lk = {};
  // L_k will always have 2**k elements since |L_k| = 2*|L_{k-1}| (and |L_3| = 2).
  Lk.reserve(pow(2, k));

  for (prufer_t S : generate_Lk_with_dup(k - 1)) {
    prufer_t S2 = S;
    // assuming that the arrays are sorted and we're adding to the back
    int max = S[S.size() - 1];

    S.push_back(max);
    S2.push_back(max + 1);
    S2.push_back(max + 1);

    Lk.push_back(S);
    Lk.push_back(S2);
  }

  return Lk;
}


// std::unordered_map<uint8_t, uint8_t> prufer_seq_to_branches(int N, prufer_t seq) {

  // def prufer_seq_to_tree(N: int, seq: List[int]) -> Dict[int, int]:
  //   # add implied final edge
  //   seq = seq.copy()
    // seq.push_back()
  //   seq.append(max(seq))
  //   # initialize things
  // std::set<uint8_t> prufer_set = std::set()
  //   prufer_set = sorted(set(seq))
  //   branches = [0] * len(prufer_set)
  //   degrees = [0] * len(prufer_set)

  //   #for each branch, sum up all the leaves that connect to it
  //   for i,node in enumerate(prufer_set):
  //       for j,n in enumerate(seq[:N]):
  //           if node == n:
  //               branches[i] += 2**j
    
  //       degrees[i] = seq.count(node)
        
    
  //   # for the branches in prufer seq, find index of parent and add its sum to parent
  //   for j,n in enumerate(seq[N:]):
  //       i = prufer_set.index(n)
  //       branches[i] += branches[j]

  //   return dict(zip(branches[::-1], degrees[::-1]))
// }

void worker(prufer_t Sk) {
  prufer_arr_t perms_of_Sk = {};
  perms_of_Sk.reserve(GPU_BATCH_SIZE);

  long n = 0;
  size_t batch_count = 0;

  // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  do {
    //TODO: is valid permutation
    perms_of_Sk[batch_count] = Sk;
    // if (Sk.size() == 5) {
    //   print_vector_u(Sk);
    // }
    
    batch_count++;

    if (batch_count == GPU_BATCH_SIZE) {
      n += batch_count;
      //   //TODO: Send off to gpu here
      // print_vector(perms_of_Sk.at(0));
      // std::cout << " batch out!" << std::endl;
      batch_count = 0;
    }
  } while (std::next_permutation(Sk.begin(), Sk.end()));

  n += batch_count;
  std::cout << "final batch out! " << n << std::endl;
  if (n == 90) {
    // for (prufer_t perm : perms_of_Sk) {
    //   std::cout << perm.size() << std::endl;
    //   print_vector_u(perm);
    // }
  }

  perm_mtx.lock();
  permutations+= n;
  perm_mtx.unlock();
}

int main(int argc, char** argv) {
  std::vector<int> primes = {2, 3, 5};
  std::vector<int> charges = {-1, -1, 1, 1};
  bool default_primes = true;
  bool default_charges = true;

  // Process user arguments for primes and charges
  for (int i = 1; i < argc; i++) {
    // checks for charges flag
    if (strcmp(argv[i], "-q") == 0 && i != argc - 1) {
      // extract array from string
      charges = extract_array(argv[++i]);
      // throws error if array is of length 0
      if (charges.size() == 0) {
        std::cerr << "Unable to extract charges array" << std::endl;
        exit(2);
      }
      default_charges = false;
    }
    // checks for the primes flag
    else if (strcmp(argv[i], "-p") == 0 && i != argc - 1) {
      // extract array from string
      primes = extract_array(argv[++i]);
      // throws error if array is of length 0
      if (primes.size() == 0) {
        std::cerr << "Unable to extract primes array" << std::endl;
        exit(2);
      }
      default_primes = false;
    } else {
      std::cerr << "Unrecognized parameter '" << argv[i] << "'" << std::endl;
    }
  }

  std::cout << (default_primes ? "Using default primes: " : "Primes: ");
  print_vector(primes);

  std::cout << (default_charges ? "Using default charges: " : "Charges: ");
  print_vector(charges);

  // Generate L_k recursively
  int k = charges.size();
  int max_size = 2 * (k - 1) - 1;
  std::cout << "Generating L_k for " << k << std::endl;
  prufer_arr_t Lk = generate_Lk(k);

  std::cout << "L_k size: " << Lk.size() << std::endl;

  // // PRINT Lk
  // for (prufer_t Sk : Lk) {  
  //   for (int n : Sk) {
  //     std::cout << n << ',';
  //   }
  //   std::cout << std::endl;
  // }

 
  // permute Lk
  // https://madhawapolkotuwa.medium.com/understanding-c-threads-a-complete-guide-7e783b22da6b
  int max_threads = std::thread::hardware_concurrency();

  std::vector<std::thread> threads;
  for (prufer_t Sk : Lk) {
    // while there are too many threads, wait for the first one to join
    while (threads.size() >= max_threads) {
      if (threads.front().joinable()) {
        threads.front().join();
        threads.erase(threads.begin());
      }
    }
    // add thread to the queue
    threads.emplace_back(worker, Sk);
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }


  // prufer_arr_t perms_of_Lk{};

  // int permutations = 0;
  // // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  // for (prufer_t Sk : Lk_dup) {
  //   do {
  //     permutations++;
  //   } while (std::next_permutation(Sk.begin(), Sk.end()));
  // }

  // int permutations = perms_of_Lk.size();


  std::cout << permutations << std::endl;
  // for (prufer_t Sk : perms_of_Lk) {
  //   // if (!is_valid_prufer(Sk)) {
  //   //   continue;
  //   // }
  //   for (int n : Sk) {
  //     std::cout << n << ",";
  //   }
  //   std::cout << "1" << std::endl;
  // }

  return 0;
}
