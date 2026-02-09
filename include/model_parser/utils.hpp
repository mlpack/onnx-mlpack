#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

/**
 * @brief Utility functions for printing data structures to the console.
 * 
 * This header provides easy-to-use operators for printing common STL containers
 * like `std::vector` and `std::map` directly to the console using `std::cout`.
 * Simply include this header, and you can print the containers as shown below.
 * 
 * ## Usage Examples:
 * 
 * ### Printing a vector:
 * std::vector<int> vec = {1, 2, 3};
 * std::cout << vec << std::endl; // Output: [1, 2, 3]
 * 
 * 
 * ### Printing a 2D vector:
 * std::vector<std::vector<int>> vec2d = {
 *     {1, 2, 3},
 *     {4, 5, 6},
 *     {7, 8, 9}
 * };
 * std::cout << vec2d << std::endl; 
 * // Output:
 * // [
 * //   [1, 2, 3]
 * //   [4, 5, 6]
 * //   [7, 8, 9]
 * // ]
 * 
 * 
 * ### Printing a map:
 * std::map<int, std::string> myMap = {{1, "one"}, {2, "two"}};
 * std::cout << myMap << std::endl; // Output: {1: one, 2: two}
 */


#include "utils_impl.hpp"
#endif