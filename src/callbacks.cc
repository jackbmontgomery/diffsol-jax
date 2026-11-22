#include "callbacks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace diffsol_jax {

namespace {

constexpr const char *kNames[kNumCallbacks] = {
    "rhs",  "rhs_jac",   "rhs_sens",          "rhs_adjoint", "rhs_sens_adjoint",
    "init", "init_sens", "init_sens_adjoint", "mass",        "mass_adjoint",
    "out",  "out_jac",   "out_sens",          "out_adjoint", "out_sens_adjoint",
};

// The callback return codes. diffsol fills the output with NaN and reports the
// callback name; the message is in `CallbackSet::error`.
constexpr int32_t kMissingVariant = 1;
constexpr int32_t kBadLayout = 2;
constexpr int32_t kPjrtError = 3;

constexpr size_t kF64 = sizeof(double);

constexpr const char *kBadInput = "an input view has an unexpected layout";

// Record the first error of the solve.
int32_t Fail(CallbackSet *set, int id, int32_t code,
             const std::string &message) {
  if (set->error.empty())
    set->error = std::string("callback '") + kNames[id] + "': " + message;
  return code;
}

// A view of an input that is constant over `P` (`x` or `p`) as a row-major `[B,
// n]` array.
bool LaneArg(const TensorView *v, ArgView *out) {
  size_t n = v->shape[0], b = v->shape[2];
  if (v->ndim != 3 || v->dtype != ScalarType_F64 || v->strides[0] != kF64)
    return false;
  if (v->shape[1] > 1 && v->strides[1] != 0)
    return false;
  if (b > 1 && v->strides[2] != n * kF64)
    return false;
  *out =
      ArgView{v->ptr, {static_cast<int64_t>(b), static_cast<int64_t>(n), 0}, 2};
  return true;
}

// A view of `v` or `y` as a row-major `[B, P, n]` array.
bool FullArg(const TensorView *v, ArgView *out) {
  size_t n = v->shape[0], p = v->shape[1], b = v->shape[2];
  if (v->ndim != 3 || v->dtype != ScalarType_F64 || v->strides[0] != kF64)
    return false;
  if (p > 1 && v->strides[1] != n * kF64)
    return false;
  if (b > 1 && v->strides[2] != n * p * kF64)
    return false;
  *out = ArgView{v->ptr,
                 {static_cast<int64_t>(b), static_cast<int64_t>(p),
                  static_cast<int64_t>(n)},
                 3};
  return true;
}

constexpr uintptr_t kAlign = 64;

// If `arg` is not 64-byte aligned, copy it into `staging` and point `arg` at
// the copy.
bool AlignArg(ArgView *arg, Staging *staging, CallbackSet *set) {
  if ((reinterpret_cast<uintptr_t>(arg->ptr) & (kAlign - 1)) == 0)
    return true;
  size_t bytes = kF64;
  for (int d = 0; d < arg->ndim; d++)
    bytes *= static_cast<size_t>(arg->dims[d]);
  void *dst = staging->Reserve(bytes);
  if (dst == nullptr)
    return false;
  std::memcpy(dst, arg->ptr, bytes);
  arg->ptr = dst;
  set->nstaged++;
  return true;
}

// Find the executable for the `P` of `y`, run it on `args` (already in the
// executable's argument order), and write `y`.
int32_t Run(CallbackSet *set, int id, ArgView *args, int nargs,
            TensorViewMut *y) {
  set->ncalls++;
  const size_t p = y->shape[1];
  Variants &v = set->variants[id];
  int slot = v.Find(p);
  if (slot < 0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "no executable for P = %zu, B = %zu; the lowering did not "
                  "compile this layout",
                  p, static_cast<size_t>(y->shape[2]));
    return Fail(set, id, kMissingVariant, buf);
  }
  Variant &variant = v.list[static_cast<size_t>(slot)];
  // an unaligned input is copied into its staging buffer; the scalars are
  // aligned already, so this is a no-op for them
  for (int i = 0; i < nargs; i++) {
    if (!AlignArg(&args[i], &set->staging[i], set))
      return Fail(set, id, kPjrtError, "cannot allocate a staging buffer");
  }
  ArgView out;
  if (!FullArg(y, &out))
    return Fail(set, id, kBadLayout, "the output view is not dense");
  size_t out_bytes = y->shape[0] * y->shape[1] * y->shape[2] * kF64;
  // the PJRT views: reuse a cached view if the pointer and the shape are the
  // same
  std::string err;
  PJRT_Buffer *buffers[kMaxArgs];
  for (int i = 0; i < nargs; i++) {
    CachedView &cv = variant.views[i];
    const ArgView &a = args[i];
    bool hit = cv.buffer != nullptr && cv.ptr == a.ptr && cv.ndim == a.ndim &&
               std::memcmp(cv.dims, a.dims,
                           sizeof(int64_t) * static_cast<size_t>(a.ndim)) == 0;
    if (!hit) {
      set->runtime->DestroyBuffer(cv.buffer);
      cv = CachedView{};
      PJRT_Buffer *buffer = set->runtime->CreateView(a, &err);
      if (buffer == nullptr)
        return Fail(set, id, kPjrtError, err);
      cv.ptr = a.ptr;
      std::memcpy(cv.dims, a.dims, sizeof(cv.dims));
      cv.ndim = a.ndim;
      cv.buffer = buffer;
      set->nviews++;
    }
    buffers[i] = cv.buffer;
  }
  if (!set->runtime->Execute(variant.exe, buffers, nargs,
                             const_cast<uint8_t *>(y->ptr), out_bytes, &err)) {
    return Fail(set, id, kPjrtError, err);
  }
  return 0;
}

// The trampolines: one for each signature, with the callback id as a template
// parameter. Each builds the argument list in the order the lowered executable
// declares its parameters (`_lower.py`), which is the order of the callback
// arguments.

template <int Id>
int32_t CbFn(const TensorView *x, const TensorView *p, double t,
             TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  ArgView args[3];
  if (!LaneArg(x, &args[0]) || !LaneArg(p, &args[1]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[2] = set->ScalarArg(0, t);
  return Run(set, Id, args, 3, y);
}

template <int Id>
int32_t CbMulFn(const TensorView *x, const TensorView *p, double t,
                const TensorView *v, TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  ArgView args[4];
  if (!LaneArg(x, &args[0]) || !LaneArg(p, &args[1]) || !FullArg(v, &args[3]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[2] = set->ScalarArg(0, t);
  return Run(set, Id, args, 4, y);
}

template <int Id>
int32_t CbInitFn(const TensorView *p, double t, TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  ArgView args[2];
  if (!LaneArg(p, &args[0]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[1] = set->ScalarArg(0, t);
  return Run(set, Id, args, 2, y);
}

template <int Id>
int32_t CbInitMulFn(const TensorView *p, double t, const TensorView *v,
                    TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  ArgView args[3];
  if (!LaneArg(p, &args[0]) || !FullArg(v, &args[2]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[1] = set->ScalarArg(0, t);
  return Run(set, Id, args, 3, y);
}

// `y` is an input and the output: the executable gets a view of `y`, and writes
// the result into the memory of `y` when it has finished. Thus the input and
// the output can be the same memory.
template <int Id>
int32_t CbMassFn(const TensorView *v, const TensorView *p, double t,
                 double beta, TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  ArgView args[5];
  if (!FullArg(v, &args[0]) || !LaneArg(p, &args[1]) || !FullArg(y, &args[4]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[2] = set->ScalarArg(0, t);
  args[3] = set->ScalarArg(1, beta);
  return Run(set, Id, args, 5, y);
}

} // namespace

Staging::~Staging() { std::free(data); }

void *Staging::Reserve(size_t bytes) {
  if (bytes <= capacity)
    return data;
  size_t rounded = (bytes + kAlign - 1) / kAlign * kAlign;
  void *fresh = std::aligned_alloc(kAlign, rounded);
  if (fresh == nullptr)
    return nullptr;
  std::free(data);
  data = fresh;
  capacity = rounded;
  return data;
}

const char *CallbackName(int id) {
  return id >= 0 && id < kNumCallbacks ? kNames[id] : "unknown";
}

int Variants::Find(size_t p) const {
  for (size_t i = 0; i < list.size(); i++) {
    if (list[i].p == p)
      return static_cast<int>(i);
  }
  return -1;
}

void CallbackSet::DestroyViews(int id, int i) {
  for (auto &cv : variants[id].list[static_cast<size_t>(i)].views) {
    runtime->DestroyBuffer(cv.buffer);
    cv = CachedView{};
  }
}

void CallbackSet::Add(int id, size_t p, PJRT_LoadedExecutable *exe) {
  Variants &v = variants[id];
  int slot = v.Find(p);
  if (slot >= 0) {
    DestroyViews(id, slot);
    Variant &existing = v.list[static_cast<size_t>(slot)];
    runtime->Destroy(existing.exe);
    existing.exe = exe;
    return;
  }
  Variant fresh;
  fresh.p = p;
  fresh.exe = exe;
  v.list.push_back(fresh);
}

bool CallbackSet::Has(int id, size_t p) const {
  return variants[id].Find(p) >= 0;
}

ArgView CallbackSet::ScalarArg(int slot, double value) {
  scalars[slot].value = value;
  return ArgView{&scalars[slot].value, {0, 0, 0}, 0};
}

void CallbackSet::Fill(OdeCallbacksDesc_f64 *d) {
  auto has = [&](int id) { return Provides(id); };
  d->rhs = has(kRhs) ? CbFn<kRhs> : nullptr;
  d->rhs_jac = has(kRhsJac) ? CbMulFn<kRhsJac> : nullptr;
  d->rhs_sens = has(kRhsSens) ? CbMulFn<kRhsSens> : nullptr;
  d->rhs_adjoint = has(kRhsAdjoint) ? CbMulFn<kRhsAdjoint> : nullptr;
  d->rhs_sens_adjoint =
      has(kRhsSensAdjoint) ? CbMulFn<kRhsSensAdjoint> : nullptr;
  d->init = has(kInit) ? CbInitFn<kInit> : nullptr;
  d->init_sens = has(kInitSens) ? CbInitMulFn<kInitSens> : nullptr;
  d->init_sens_adjoint =
      has(kInitSensAdjoint) ? CbInitMulFn<kInitSensAdjoint> : nullptr;
  d->mass = has(kMass) ? CbMassFn<kMass> : nullptr;
  d->mass_adjoint = has(kMassAdjoint) ? CbMassFn<kMassAdjoint> : nullptr;
  d->out = has(kOut) ? CbFn<kOut> : nullptr;
  d->out_jac = has(kOutJac) ? CbMulFn<kOutJac> : nullptr;
  d->out_sens = has(kOutSens) ? CbMulFn<kOutSens> : nullptr;
  d->out_adjoint = has(kOutAdjoint) ? CbMulFn<kOutAdjoint> : nullptr;
  d->out_sens_adjoint =
      has(kOutSensAdjoint) ? CbMulFn<kOutSensAdjoint> : nullptr;
  d->user_data = this;
  d->free_user_data = nullptr;
}

std::string CallbackSet::TakeError() {
  std::string e;
  e.swap(error);
  return e;
}

void CallbackSet::DestroyAll() {
  for (int id = 0; id < kNumCallbacks; id++) {
    Variants &v = variants[id];
    for (size_t i = 0; i < v.list.size(); i++) {
      DestroyViews(id, static_cast<int>(i));
      runtime->Destroy(v.list[i].exe);
    }
    v.list.clear();
  }
}

} // namespace diffsol_jax
