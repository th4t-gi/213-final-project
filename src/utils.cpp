#include "utils.h"
#include <algorithm>

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

void print_vector_u(std::vector<uint8_t> v) {
//   std::cout << "hello?" << v.size() << std::endl;
  for (uint8_t entry : v) {
    printf("%d,", entry);
  }
  std::cout << std::endl;
}

void print_vector(std::vector<int> v) {
  for (int entry : v) {
    std::cout << entry << ",";
  }
  std::cout << std::endl;
}
void print_vector(std::vector<uint16_t> v) {
  for (uint16_t entry : v) {
    std::cout << entry << ",";
  }
  std::cout << std::endl;
}