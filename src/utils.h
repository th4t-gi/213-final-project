#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <unordered_map>

typedef std::vector<uint8_t> prufer_t;
typedef std::vector<prufer_t> prufer_arr_t;


typedef uint16_t code_t;
typedef uint8_t label_size_t;
typedef uint8_t degree_t;

typedef std::unordered_map<code_t, degree_t> tree_map_t;

std::vector<int> extract_array(char* str);
void print_vector_u(std::vector<uint8_t> v);
void print_vector(std::vector<int> v);
void print_vector(std::vector<uint16_t> v);

tree_map_t prufer_seq_to_branches(int N, prufer_t seq);