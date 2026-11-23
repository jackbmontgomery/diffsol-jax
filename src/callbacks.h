#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backend.h"
#include "diffsol.h"

namespace diffsol_jax {

enum CallbackId : int {
  kRhs = 0,
  kRhsJac,
  kRhsSens,
  kRhsAdjoint,
  kRhsSensAdjoint,
  kInit,
  kInitSens,
  kInitSensAdjoint,
  kMass,
  kMassAdjoint,
  kOut,
  kOutJac,
  kOutSens,
  kOutAdjoint,
  kOutSensAdjoint,
  kNumCallbacks,
};

const char *CallbackName(int id);

struct CachedView {
  const void *ptr = nullptr;
  int64_t dims[kMaxDims] = {};
  int ndim = -1;
  PJRT_Buffer *buffer = nullptr;
};

struct Variant {
  size_t p = 0;
  PJRT_LoadedExecutable *exe = nullptr;
  CachedView views[kMaxArgs];
};

// The executables of one callback, one for each `P`.
struct Variants {
  std::vector<Variant> list;
  int Find(size_t p) const;
};

struct Staging {
  Staging() = default;
  Staging(const Staging &) = delete;
  Staging &operator=(const Staging &) = delete;
  ~Staging();
  void *Reserve(size_t bytes, size_t align);
  void *data = nullptr;
  size_t capacity = 0;
};

struct alignas(64) AlignedScalar {
  union {
    double f64;
    float f32;
  } value = {0.0};
};

// The `user_data` of the callbacks of one `OdeWrapper` (one model, one `B`).
struct CallbackSet {
  Backend *backend = nullptr;
  Dtype dtype = Dtype::kF64;
  size_t nstates = 0;
  size_t nparams = 0;
  size_t nout = 0;
  size_t nbatch = 0;
  std::array<Variants, kNumCallbacks> variants;
  AlignedScalar scalars[2];
  Staging staging[kMaxArgs];
  uint64_t ncalls = 0;
  uint64_t nstaged = 0;
  uint64_t nviews = 0;
  std::string error;

  void Add(int id, size_t p, PJRT_LoadedExecutable *exe);
  bool Has(int id, size_t p) const;
  bool Provides(int id) const { return !variants[id].list.empty(); }
  void Fill(OdeCallbacksDesc_f64 *desc);
  void Fill(OdeCallbacksDesc_f32 *desc);
  ArgView ScalarArg(int slot, double value);
  std::string TakeError();
  void DestroyAll();
  void DestroyViews(int id, int i);
};

} // namespace diffsol_jax
