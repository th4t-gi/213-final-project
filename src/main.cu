#include <algorithm>
#include <mutex>
#include <thread>
#include <cmath>

#include "worker.h"

// GLOBAL PROGRAM PARAMETERS
// primarily related to concurrency/parralellism aspects of program
#define GPU_BATCH_SIZE 10
#define MAX_THREADS 8
#define DEFAULT_BETA_STEP 0.1

global_params_t params;

/**
 * Recursively generates "prufer sets" (i.e. prufer codes for unlabeled trees) to be s
 *
 * \param k number of leaves of trees
 */
prufer_arr_t generate_Lk_with_dup(int k) {
  // BASE CASE:
  if (k <= 2) {
    prufer_arr_t base{{1}};
    return base;
  }

  // RECURSIVE CASE:
  prufer_arr_t Lk = {};
  // L_k will always have 2**(k-3) elements since |L_k| = 2*|L_{k-1}| (and |L_3| = 2).
  Lk.reserve(2UL << (k-3));

  for (prufer_t S : generate_Lk_with_dup(k - 1)) {
    prufer_t S2 = S;
    // assuming that the arrays are sorted and we're adding to the back, then max elt is always last index
    int max = S[S.size() - 1];

    S.push_back(max);
    S2.push_back(max + 1);
    S2.push_back(max + 1);

    Lk.push_back(S);
    Lk.push_back(S2);
  }

  return Lk;
}

/**
 * Recursively generates "prufer sets" (i.e. prufer codes for unlabeled trees) to be s
 *
 * \param params struct to save processed user inputs to
 * \param argc number of arguments in argv
 * \param argv array of arguments given by main()
 */
void process_params(global_params_t* params, int argc, char** argv) {
  // initializes default parameters
  params->primes = {2, 3, 5};
  params->charges = {-1, -1, 1, 1};
  params->beta_step = DEFAULT_BETA_STEP;
  bool default_primes = true;
  bool default_charges = true;
  bool default_step = true;

  // Iterates through elements of argv to find primes, charges, and beta_step
  for (int i = 1; i < argc; i++) {

    //prints help message and exits
    if (strcmp(argv[i], "--help") == 0) {
      printf("this is a help message!");
      exit(0);
    }
    // checks for charges flag
    else if (strcmp(argv[i], "-q") == 0 && i != argc - 1) {
      // extract array from next argument 
      params->charges = extract_array<double>(argv[++i]);
      // throws error if array is of length 0
      if (params->charges.size() == 0) {
        std::cerr << "Unable to extract charges array" << std::endl;
        exit(2);
      }
      default_charges = false;
    }
    // checks for the primes flag
    else if (strcmp(argv[i], "-p") == 0 && i != argc - 1) {
      // extract array from next argument 
      params->primes = extract_array<int>(argv[++i]);
      // throws error if array is of length 0
      if (params->primes.size() == 0) {
        std::cerr << "Unable to extract primes array" << std::endl;
        exit(2);
      }
      default_primes = false;
    }
    // checks for beta_step flag
    else if (strcmp(argv[i], "-b") == 0 && i != argc - 1) {
      // extract beta_step from next argument 
      params->beta_step = atof(argv[++i]);
      default_step = false;
    } else {
      std::cerr << "Unrecognized parameter '" << argv[i] << "'" << std::endl;
    }
  }

  //Computes Beta critical
  int q_max = *std::max_element(params->charges.begin(), params->charges.end());
  int q_min = *std::min_element(params->charges.begin(), params->charges.end());
  double beta_c = 1.0/abs(q_max*q_min);

  params->beta_count = ceil(beta_c/params->beta_step);
  params->partition_values = (double*) malloc(sizeof(double)*params->primes.size()*params->beta_count);


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

  // Generates L_k recursively
  int k = params.charges.size();
  std::cout << "Generating L_k for " << k << std::endl;
  prufer_arr_t Lk = generate_Lk_with_dup(k);

  std::cout << "L_k size: " << Lk.size() << std::endl;

  // generates a thread for each element of Lk and concurrently generates permutations of that element
  // CITATION: https://madhawapolkotuwa.medium.com/understanding-c-threads-a-complete-guide-7e783b22da6b
  // This article helped me figure out how to use the std library thread interface.
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
    threads.emplace_back(worker, k, Sk, &params, GPU_BATCH_SIZE);
  }

  // joins all the threads
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  std::cout << "total permutations: " << params.permutations << std::endl;

  //prints the outputs matrix 
  for (int i = 0; i < params.primes.size(); i++) {
    for (int j = 0; j < params.beta_count; j++) {
      int idx = i*params.primes.size() + j;
      printf("p: %d, b: %lf, Z_I(b): %lf\n", params.primes[i], params.beta_step*j, params.partition_values[idx]);
    }
    printf("\n");
  }

  free(params.partition_values);

  return 0;
}
