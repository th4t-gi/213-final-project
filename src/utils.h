#pragma once

#include <vector>

typedef std::vector<uint8_t> prufer_t;
typedef std::vector<prufer_t> prufer_arr_t;


std::vector<int> extract_array(char* str);
void print_vector_u(std::vector<uint8_t> v);
void print_vector(std::vector<int> v);