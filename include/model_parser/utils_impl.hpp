#include "utils.hpp"

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
    for (int i = 0; i < v.size(); i++)
      std::copy(
          v.at(i).begin(), v.at(i).end(), std::ostream_iterator<T>(out, ","));
    out << "\b]";
  }
  return out;
}

template <typename T1, typename T2>
std::ostream &
operator<<(std::ostream &out, const std::map<T1, T2> &m)
{
  out << "{\n";
  for (auto it : m)
  {
    out << "  {" << it.first << " : " << it.second << "}\n";
  }
  out << "\b}";
  return out;
}
