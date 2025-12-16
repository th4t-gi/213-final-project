#include <algorithm>
#include <mutex>
#include <thread>
#include <cmath>

#include "utils.h"
#include "kernel.h"

#define GPU_BATCH_SIZE 100
#define MAX_THREADS 1

//https://www.geeksforgeeks.org/cpp/std-mutex-in-cpp/
std::mutex perm_mtx;
global_params_t params;


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


void process_params(global_params_t* params, int argc, char** argv) {
  // initializes default parameters
  params->primes = {2, 3, 5};
  params->charges = {-1, -1, 1, 1};
  params->beta_step = 0.01;
  bool default_primes = true;
  bool default_charges = true;
  bool default_step = true;

  // Process user arguments for primes and charges
  for (int i = 1; i < argc; i++) {

    if (strcmp(argv[i], "--help") == 0) {
      printf("this is a help message!");
      exit(0);
    }
    // checks for charges flag
    else if (strcmp(argv[i], "-q") == 0 && i != argc - 1) {
      // extract array from string
      params->charges = extract_array(argv[++i]);
      // throws error if array is of length 0
      if (params->charges.size() == 0) {
        std::cerr << "Unable to extract charges array" << std::endl;
        exit(2);
      }
      default_charges = false;
    }
    // checks for the primes flag
    else if (strcmp(argv[i], "-p") == 0 && i != argc - 1) {
      // extract array from string
      params->primes = extract_array(argv[++i]);
      // throws error if array is of length 0
      if (params->primes.size() == 0) {
        std::cerr << "Unable to extract primes array" << std::endl;
        exit(2);
      }
      default_primes = false;
    }
    // checks for beta_step flag
    else if (strcmp(argv[i], "-b") == 0 && i != argc - 1) {
      params->beta_step = atof(argv[++i]);
      default_step = false;
    } else {
      std::cerr << "Unrecognized parameter '" << argv[i] << "'" << std::endl;
    }
  }

  //Computes Beta critical
  int q_max = *std::max_element(params->charges.begin(), params->charges.end());
  int q_min = *std::min_element(params->charges.begin(), params->charges.end());
  params->beta_c = 1.0/abs(q_max*q_min);

  std::cout << (default_primes ? "Using default primes: " : "Primes: ");
  print_vector(params->primes);

  std::cout << (default_charges ? "Using default charges: " : "Charges: ");
  print_vector(params->charges);

  std::cout << (default_step ? "Using default beta step: " : "Beta Step: ");
  printf("%.4f\n", params->beta_step);
}



int main(int argc, char** argv) {
  // processes the user input and saves to global struct params
  process_params(&params, argc, argv);

  // Generate L_k recursively
  int k = params.charges.size();
  int max_size = 2 * (k - 1) - 1;
  std::cout << "Generating L_k for " << k << std::endl;
  prufer_arr_t Lk = generate_Lk(k);

  std::cout << "L_k size: " << Lk.size() << std::endl;

  // PRINTS Lk
  // for (prufer_t Sk : Lk) {  
  //   for (int n : Sk) {
  //     std::cout << n << ',';
  //   }
  //   std::cout << std::endl;
  // }

  // generates a thread for each element of Lk and concurrently generates permutations of that element
  // https://madhawapolkotuwa.medium.com/understanding-c-threads-a-complete-guide-7e783b22da6b
  int max_threads = std::min((unsigned int) MAX_THREADS, std::thread::hardware_concurrency());
  std::cout << "Max threads: " << max_threads << std::endl;

  // creates thread pool with max_threads number of threads.
  std::vector<std::thread> threads;
  threads.reserve(max_threads);

  for (prufer_t Sk : Lk) {
    // while there are too many threads, wait for the first one to join
    while (threads.size() >= max_threads) {
      // std::cout << "waiting...";
      if (threads.front().joinable()) {
        threads.front().join();
        threads.erase(threads.begin());
      }
    }
    // add thread to the queue
    threads.emplace_back(worker, k, Sk);
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      // std::cout << "joining thread!" << std::endl;
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
