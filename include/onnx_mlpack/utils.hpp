/**
 * @file utils.hpp
 * @author Kumar Utkarsh
 *
 * Utilities for printing vectors.
 */
#ifndef ONNX_MLPACK_UTILS_HPP
#define ONNX_MLPACK_UTILS_HPP

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v)
{
  if (!v.empty())
  {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b]";
  }
  return out;
}

template <typename T>
std::ostream &
operator<<(std::ostream &out, const std::vector<std::vector<T>> &v)
{
  if (!v.empty())
  {
    out << '[';
    for (size_t i = 0; i < v.size(); i++)
    {
      std::copy(v.at(i).begin(), v.at(i).end(),
          std::ostream_iterator<T>(out, ","));
    }

    out << "\b]";
  }
  return out;
}

template <typename T1, typename T2>
std::ostream &
operator<<(std::ostream &out, const std::map<T1, T2> &m)
{
  out << "{\n";
  for (const std::pair<T1, T2> it : m)
  {
    out << "  {" << it.first << " : " << it.second << "}\n";
  }
  out << "\b}";
  return out;
}

#endif
