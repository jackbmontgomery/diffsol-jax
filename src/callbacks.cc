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
bool LaneArg(const TensorView *v, Dtype dtype, ArgView *out) {
  const size_t width = ElementSize(dtype);
  size_t n = v->shape[0], b = v->shape[2];
  if (v->ndim != 3 || v->dtype != DiffsolType(dtype) || v->strides[0] != width)
    return false;
  if (v->shape[1] > 1 && v->strides[1] != 0)
    return false;
  if (b > 1 && v->strides[2] != n * width)
    return false;
  *out = ArgView{
      v->ptr, {static_cast<int64_t>(b), static_cast<int64_t>(n), 0}, 2, dtype};
  return true;
}

// A view of `v` or `y` as a row-major `[B, P, n]` array.
bool FullArg(const TensorView *v, Dtype dtype, ArgView *out) {
  const size_t width = ElementSize(dtype);
  size_t n = v->shape[0], p = v->shape[1], b = v->shape[2];
  if (v->ndim != 3 || v->dtype != DiffsolType(dtype) || v->strides[0] != width)
    return false;
  if (p > 1 && v->strides[1] != n * width)
    return false;
  if (b > 1 && v->strides[2] != n * p * width)
    return false;
  *out = ArgView{v->ptr,
                 {static_cast<int64_t>(b), static_cast<int64_t>(p),
                  static_cast<int64_t>(n)},
                 3,
                 dtype};
  return true;
}

// If `arg` does not meet the backend's alignment, copy it into `staging` and
// point `arg` at the copy.
bool AlignArg(ArgView *arg, Staging *staging, CallbackSet *set) {
  const size_t align = set->backend->view_alignment();
  if ((reinterpret_cast<uintptr_t>(arg->ptr) & (align - 1)) == 0)
    return true;
  size_t bytes = ElementSize(arg->dtype);
  for (int d = 0; d < arg->ndim; d++)
    bytes *= static_cast<size_t>(arg->dims[d]);
  void *dst = staging->Reserve(bytes, align);
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
  if (!FullArg(y, set->dtype, &out))
    return Fail(set, id, kBadLayout, "the output view is not dense");
  size_t out_bytes =
      y->shape[0] * y->shape[1] * y->shape[2] * ElementSize(set->dtype);
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
      set->backend->DestroyBuffer(cv.buffer);
      cv = CachedView{};
      PJRT_Buffer *buffer = set->backend->MakeView(a, &err);
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
  if (!set->backend->Execute(variant.exe, buffers, nargs,
                             const_cast<uint8_t *>(y->ptr), out_bytes, &err)) {
    return Fail(set, id, kPjrtError, err);
  }
  return 0;
}

// The trampolines: one for each signature, with the scalar type and the
// callback id as template parameters. `T` is only the type of the scalar
// arguments diffsol passes by value; the array views take their dtype from the
// set, and `ScalarArg` stores the value in that dtype. Each builds the argument
// list in the order the lowered executable declares its parameters
// (`_lower.py`), which is the order of the callback arguments.

template <typename T, int Id>
int32_t CbFn(const TensorView *x, const TensorView *p, T t, TensorViewMut *y,
             void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  const Dtype d = set->dtype;
  ArgView args[3];
  if (!LaneArg(x, d, &args[0]) || !LaneArg(p, d, &args[1]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[2] = set->ScalarArg(0, t);
  return Run(set, Id, args, 3, y);
}

template <typename T, int Id>
int32_t CbMulFn(const TensorView *x, const TensorView *p, T t,
                const TensorView *v, TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  const Dtype d = set->dtype;
  ArgView args[4];
  if (!LaneArg(x, d, &args[0]) || !LaneArg(p, d, &args[1]) ||
      !FullArg(v, d, &args[3]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[2] = set->ScalarArg(0, t);
  return Run(set, Id, args, 4, y);
}

template <typename T, int Id>
int32_t CbInitFn(const TensorView *p, T t, TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  const Dtype d = set->dtype;
  ArgView args[2];
  if (!LaneArg(p, d, &args[0]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[1] = set->ScalarArg(0, t);
  return Run(set, Id, args, 2, y);
}

template <typename T, int Id>
int32_t CbInitMulFn(const TensorView *p, T t, const TensorView *v,
                    TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  const Dtype d = set->dtype;
  ArgView args[3];
  if (!LaneArg(p, d, &args[0]) || !FullArg(v, d, &args[2]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[1] = set->ScalarArg(0, t);
  return Run(set, Id, args, 3, y);
}

// `y` is an input and the output: the executable gets a view of `y`, and writes
// the result into the memory of `y` when it has finished. Thus the input and
// the output can be the same memory.
template <typename T, int Id>
int32_t CbMassFn(const TensorView *v, const TensorView *p, T t, T beta,
                 TensorViewMut *y, void *ud) {
  auto *set = static_cast<CallbackSet *>(ud);
  const Dtype d = set->dtype;
  ArgView args[5];
  if (!FullArg(v, d, &args[0]) || !LaneArg(p, d, &args[1]) ||
      !FullArg(y, d, &args[4]))
    return Fail(set, Id, kBadLayout, kBadInput);
  args[2] = set->ScalarArg(0, t);
  args[3] = set->ScalarArg(1, beta);
  return Run(set, Id, args, 5, y);
}

} // namespace

Staging::~Staging() { std::free(data); }

void *Staging::Reserve(size_t bytes, size_t align) {
  if (bytes <= capacity)
    return data;
  size_t rounded = (bytes + align - 1) / align * align;
  void *fresh = std::aligned_alloc(align, rounded);
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
    backend->DestroyBuffer(cv.buffer);
    cv = CachedView{};
  }
}

void CallbackSet::Add(int id, size_t p, PJRT_LoadedExecutable *exe) {
  Variants &v = variants[id];
  int slot = v.Find(p);
  if (slot >= 0) {
    DestroyViews(id, slot);
    Variant &existing = v.list[static_cast<size_t>(slot)];
    backend->Destroy(existing.exe);
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
  AlignedScalar &s = scalars[slot];
  if (dtype == Dtype::kF32)
    s.value.f32 = static_cast<float>(value);
  else
    s.value.f64 = value;
  return ArgView{&s.value, {0, 0, 0}, 0, dtype};
}

namespace {

// Point `d` at the trampolines of the callbacks that have an executable. The
// two diffsol descriptor structs have the same field names, so one template
// fills either; `T` is the scalar type of the matching one.
template <typename T, typename Desc> void FillDesc(CallbackSet *set, Desc *d) {
  auto has = [&](int id) { return set->Provides(id); };
  d->rhs = has(kRhs) ? CbFn<T, kRhs> : nullptr;
  d->rhs_jac = has(kRhsJac) ? CbMulFn<T, kRhsJac> : nullptr;
  d->rhs_sens = has(kRhsSens) ? CbMulFn<T, kRhsSens> : nullptr;
  d->rhs_adjoint = has(kRhsAdjoint) ? CbMulFn<T, kRhsAdjoint> : nullptr;
  d->rhs_sens_adjoint =
      has(kRhsSensAdjoint) ? CbMulFn<T, kRhsSensAdjoint> : nullptr;
  d->init = has(kInit) ? CbInitFn<T, kInit> : nullptr;
  d->init_sens = has(kInitSens) ? CbInitMulFn<T, kInitSens> : nullptr;
  d->init_sens_adjoint =
      has(kInitSensAdjoint) ? CbInitMulFn<T, kInitSensAdjoint> : nullptr;
  d->mass = has(kMass) ? CbMassFn<T, kMass> : nullptr;
  d->mass_adjoint = has(kMassAdjoint) ? CbMassFn<T, kMassAdjoint> : nullptr;
  d->out = has(kOut) ? CbFn<T, kOut> : nullptr;
  d->out_jac = has(kOutJac) ? CbMulFn<T, kOutJac> : nullptr;
  d->out_sens = has(kOutSens) ? CbMulFn<T, kOutSens> : nullptr;
  d->out_adjoint = has(kOutAdjoint) ? CbMulFn<T, kOutAdjoint> : nullptr;
  d->out_sens_adjoint =
      has(kOutSensAdjoint) ? CbMulFn<T, kOutSensAdjoint> : nullptr;
  d->user_data = set;
  d->free_user_data = nullptr;
}

} // namespace

void CallbackSet::Fill(OdeCallbacksDesc_f64 *d) { FillDesc<double>(this, d); }

void CallbackSet::Fill(OdeCallbacksDesc_f32 *d) { FillDesc<float>(this, d); }

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
      backend->Destroy(v.list[i].exe);
    }
    v.list.clear();
  }
}

} // namespace diffsol_jax
