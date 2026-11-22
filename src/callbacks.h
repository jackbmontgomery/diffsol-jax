// The `OdeCallbacksDesc_f64` function pointers of diffsol-c, with PJRT
// executables behind them.
//
// Lane layout (from the diffsol-c test `lane_layout_rule`): diffsol calls a
// callback with views of the shape `[n, P, B]` (column-major). The `B` axis is
// never broadcast. If `P > 1`, the inputs `x` and `p` are broadcast on `P`
// (stride 0) and `v` is not. Thus, as row-major arrays with no copy:
//   x: [B, nstates]   p: [B, nparams]   v: [B, P, nv]   y: [B, P, ny]   t,
//   beta: []
// and the only variable of the layout is `P`. Each callback has one executable
// for each `P`.
//
// The views carry the solver's dtype (`CallbackSet::dtype`), and the
// trampolines are templated on the scalar type: `Fill` has an overload for each
// of the two diffsol descriptor structs, which have the same field names.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backend.h"
#include "diffsol.h"

namespace diffsol_jax {

// The callbacks that the prototype supports (no `root` and `reset`: diffsol-c
// does not allow them with more than one lane). The values are the same as
// `CallbackId` in Python.
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

// A PJRT view of one argument, kept between calls. diffsol often gives the same
// pointers again
// (`p` for the whole solve, the scalar slots and the staging buffers always),
// so the view is made only when the pointer or the shape changes. A view is
// only a pointer and a shape: the client reads the memory at execute time
// (spike), so a view of memory that diffsol has freed is harmless while it is
// not used, and it is used only in a call where the pointer is valid.
struct CachedView {
  const void *ptr = nullptr;
  int64_t dims[kMaxDims] = {};
  int ndim = -1;
  PJRT_Buffer *buffer = nullptr;
};

// The executable of one callback for one `P`, with its own views: each `P`
// gives different shapes. There are at most 3 in practice (`P == 1`, `P ==
// nparams` and `P == ndirections`).
struct Variant {
  size_t p = 0;
  PJRT_LoadedExecutable *exe = nullptr;
  CachedView views[kMaxArgs];
};

// The executables of one callback, one for each `P`.
struct Variants {
  std::vector<Variant> list;
  // The index of the executable for `p`, or -1.
  int Find(size_t p) const;
};

// An aligned buffer that grows only when it must, for an input whose pointer
// does not meet `Backend::view_alignment()`: it is copied in here first.
struct Staging {
  Staging() = default;
  Staging(const Staging &) = delete;
  Staging &operator=(const Staging &) = delete;
  ~Staging();
  // A buffer of at least `bytes` bytes, aligned to `align`.
  void *Reserve(size_t bytes, size_t align);
  void *data = nullptr;
  size_t capacity = 0;
};

// A scalar argument, alone in its own aligned block for the same reason, held
// in the solver's dtype so that a view of it has that dtype.
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
  // The scalar arguments `t` and `beta`: PJRT gets a view of them, so they need
  // an address.
  AlignedScalar scalars[2];
  // One staging buffer for each argument position.
  Staging staging[kMaxArgs];
  // Statistics: the number of callback calls, of inputs copied because of
  // alignment, and of PJRT views made (the misses of the view cache).
  uint64_t ncalls = 0;
  uint64_t nstaged = 0;
  uint64_t nviews = 0;
  // The first error of a callback since the last `TakeError`.
  std::string error;

  // Add the executable of callback `id` for `p` lanes. Replace an existing one.
  void Add(int id, size_t p, PJRT_LoadedExecutable *exe);
  bool Has(int id, size_t p) const;
  // Whether callback `id` has any executable, so that `Fill` gives it to
  // diffsol.
  bool Provides(int id) const { return !variants[id].list.empty(); }
  // Set the function pointers of the callbacks that have at least one
  // executable. The overload must match `dtype`: the trampolines of an f32
  // solver take their scalars as `float`.
  void Fill(OdeCallbacksDesc_f64 *desc);
  void Fill(OdeCallbacksDesc_f32 *desc);
  // Store scalar argument `slot` (`t` or `beta`) and return a view of it. PJRT
  // gets a view, so the value needs an address of its own.
  ArgView ScalarArg(int slot, double value);
  std::string TakeError();
  void DestroyAll();
  // Destroy the cached views of executable `i` of callback `id`.
  void DestroyViews(int id, int i);
};

} // namespace diffsol_jax
