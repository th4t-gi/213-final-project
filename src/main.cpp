#include <array>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

typedef std::vector<uint8_t> prufer_t;
typedef std::vector<prufer_t> prufer_arr_t;

#define BATCH_SIZE 100000

//https://www.geeksforgeeks.org/cpp/std-mutex-in-cpp/
std::mutex perm_mtx;
long permutations = 0;

std::vector<int> extract_array(char* str) {
  std::vector<int> arr;

  // CITATION: https://stackoverflow.com/a/26228023 and `man strsep`
  char* token;
  while ((token = strsep(&str, ","))) {
    int n = atoi(token);
    if (n != 0) arr.push_back(n);
  }

  return arr;
}

prufer_arr_t generate_Lk(int k) {
  // BASE CASE:
  if (k <= 2) {
    prufer_arr_t base{{1}};
    return base;
  }

  // RECURSIVE CASE:
  prufer_arr_t Lk = {};
  // L_k will always have 2**k elements since |L_k| = 2*|L_{k-1}| (and |L_3| = 2).
  Lk.reserve(pow(2, k));

  for (prufer_t S : generate_Lk(k - 1)) {
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


void worker(prufer_t Sk) {
  // std::array<prufer_t, BATCH_SIZE> perms_of_Sk;
  
  long n = 0;
  size_t batch_count = 0;
  // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  do {
    n++;
    // perms_of_Sk[batch_count] = Sk;
    
    // batch_count++;

    // if (batch_count == BATCH_SIZE) {
    //   n += batch_count;
    //   //TODO: Send off to gpu here
    //   std::cout << "batch out!" << std::endl;
    //   batch_count = 0;
    // }
  } while (std::next_permutation(Sk.begin(), Sk.end()));


  // int n = perms_of_Sk.size();
  std::cout << n << std::endl;

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
  for (int p : primes) {
    std::cout << p << ",";
  }
  std::cout << std::endl;

  std::cout << (default_charges ? "Using default charges: " : "Charges: ");
  for (int q : charges) {
    std::cout << q << ",";
  }
  std::cout << std::endl;

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
  // std::vector<std::thread> threads;
  // for (prufer_t Sk : Lk) {
  //   threads.emplace_back(worker, Sk);
  // }

  // for (auto& thread : threads) {
  //   if (thread.joinable()) {
  //     thread.join();
  //   }
  // }


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
