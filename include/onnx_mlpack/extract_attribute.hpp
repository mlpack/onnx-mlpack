/**
 * @file extract_attribute.hpp
 * @author Ryan Curtin
 *
 * Given the name of an attribute, extract its value from the given node and
 * return a bool indicating whether the operation was successful.
 */
#ifndef ONNX_MLPACK_EXTRACT_ATTRIBUTE_HPP
#define ONNX_MLPACK_EXTRACT_ATTRIBUTE_HPP

#include <armadillo>
#include <onnx/onnx_pb.h>

namespace onnx_mlpack {

/**
 * Get the ONNX AttributeType corresponding to the given type.
 */
template<typename eT>
inline onnx::AttributeProto_AttributeType GetAttributeType()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_UNDEFINED;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<float>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<double>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<int>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<std::string>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<std::vector<float>>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<std::vector<double>>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<std::vector<int>>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS;
}

template<>
inline onnx::AttributeProto_AttributeType GetAttributeType<std::vector<std::string>>()
{
  return onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS;
}

/**
 * Given an attribute that is known to have the correct type (via
 * GetAttributeType() above), set the value `val` to whatever the value of the
 * attribute is.
 */
template<typename eT>
inline bool ExtractAttribute(const onnx::AttributeProto& a, eT& val)
{
  return false; // We don't know how to extract the default case.
}

template<>
inline bool ExtractAttribute<float>(const onnx::AttributeProto& a, float& val)
{
  if (!a.has_f())
    return false;

  val = a.f();
  return true;
}

template<>
inline bool ExtractAttribute<double>(const onnx::AttributeProto& a, double& val)
{
  if (!a.has_f())
    return false;

  val = (double) a.f();
  return true;
}

template<>
inline bool ExtractAttribute<int>(const onnx::AttributeProto& a, int& val)
{
  if (!a.has_i())
    return false;

  val = a.i();
  return true;
}

template<>
inline bool ExtractAttribute<std::string>(const onnx::AttributeProto& a,
                                          std::string& val)
{
  if (!a.has_s())
    return false;

  val = a.s();
  return true;
}

template<>
inline bool ExtractAttribute<std::vector<float>>(const onnx::AttributeProto& a,
                                                 std::vector<float>& val)
{
  if (a.floats_size() == 0)
    return false;

  val.resize(a.floats_size());
  for (size_t i = 0; i < a.floats_size(); ++i)
    val[i] = a.floats(i);

  return true;
}

template<>
inline bool ExtractAttribute<std::vector<double>>(const onnx::AttributeProto& a,
                                                  std::vector<double>& val)
{
  if (a.floats_size() == 0)
    return false;

  val.resize(a.floats_size());
  for (size_t i = 0; i < a.floats_size(); ++i)
    val[i] = (double) a.floats(i);

  return true;
}

template<>
inline bool ExtractAttribute<std::vector<int>>(const onnx::AttributeProto& a,
                                               std::vector<int>& val)
{
  if (a.ints_size() == 0)
    return false;

  val.resize(a.ints_size());
  for (size_t i = 0; i < a.ints_size(); ++i)
    val[i] = a.ints(i);

  return true;
}

template<>
inline bool ExtractAttribute<std::vector<std::string>>(
    const onnx::AttributeProto& a,
    std::vector<std::string>& val)
{
  if (a.strings_size() == 0)
    return false;

  val.resize(a.strings_size());
  for (size_t i = 0; i < a.strings_size(); ++i)
    val[i] = a.strings(i);

  return true;
}

/**
 * Given the name of an attribute, extract its value from the given node and
 * return a bool indicating whether the operation was successful.  The operation
 * is only considered unsuccessful if the type of the attribute does not match
 * the given type eT.  That is, `true` is returned and the value of `val` is
 * left unchanged if the attribute is not found in `node`.
 */
template<typename eT = double>
bool ExtractAttribute(const onnx::NodeProto& node,
                      const std::string& attributeName,
                      eT& val)
{
  for (size_t i = 0; i < node.attribute_size(); ++i)
  {
    const onnx::AttributeProto& a = node.attribute(i);
    if (a.has_name() && a.name() == attributeName)
    {
      // Check the type of the attribute.
      if (a.type() != GetAttributeType<eT>())
        return false;

      // Now extract the attribute.
      return ExtractAttribute<eT>(a, val);
    }
  }

  return true;
}

} // namespace onnx_mlpack

#endif
