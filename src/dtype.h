// The scalar type of a solver, and its three mappings: the size of a value, the
// PJRT buffer type, and the diffsol-c `ScalarType`.
//
// This is the only header that spells out the difference between f32 and f64.
// Everything else carries a `Dtype` and asks here.
#pragma once

#include <cstddef>
#include <cstdint>

#include "diffsol.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace diffsol_jax {

// The values are diffsol-c's `ScalarType`, so a `Dtype` crosses the C boundary
// as it is.
enum class Dtype : int32_t {
  kF32 = ScalarType_F32,
  kF64 = ScalarType_F64,
};

constexpr bool IsDtype(int32_t value) {
  return value == ScalarType_F32 || value == ScalarType_F64;
}

constexpr size_t ElementSize(Dtype dtype) {
  return dtype == Dtype::kF32 ? sizeof(float) : sizeof(double);
}

constexpr PJRT_Buffer_Type PjrtType(Dtype dtype) {
  return dtype == Dtype::kF32 ? PJRT_Buffer_Type_F32 : PJRT_Buffer_Type_F64;
}

constexpr ScalarType DiffsolType(Dtype dtype) {
  return dtype == Dtype::kF32 ? ScalarType_F32 : ScalarType_F64;
}

constexpr const char *DtypeName(Dtype dtype) {
  return dtype == Dtype::kF32 ? "f32" : "f64";
}

// The `Dtype` of a C++ scalar type, for code templated on it.
template <typename T> struct DtypeOf;
template <> struct DtypeOf<float> {
  static constexpr Dtype value = Dtype::kF32;
};
template <> struct DtypeOf<double> {
  static constexpr Dtype value = Dtype::kF64;
};

} // namespace diffsol_jax
