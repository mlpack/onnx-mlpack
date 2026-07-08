/**
 * @file log.hpp
 * @author Ryan Curtin
 *
 * Logging infrastructure for different logging levels.
 */
#ifndef ONNX_MLPACK_LOG_HPP
#define ONNX_MLPACK_LOG_HPP

namespace onnx_mlpack {

/**
 * Log the given message, if `level >= threshold`.
 *
 * Intended log levels:
 *
 * 0. Logs nothing.
 * 1. Basic output about subgraph matching results.
 * 2. Detailed output about subgraph matching results.
 */
template<typename T>
void Log(const size_t level, const size_t threshold, const T& t)
{
  if (level >= threshold)
    std::cout << t;
}

/**
 * Log the given messages, if `level > threshold`.
 */
template<typename T, typename... Args>
void Log(const size_t level, const size_t threshold, const T& t, Args&&... ts)
{
  if (level > threshold)
  {
    Log(level, threshold, t);
    Log(level, threshold, ts...);
  }
}

} // namespace onnx_mlpack

#endif
