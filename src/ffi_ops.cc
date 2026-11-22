// The diffsol-c solve functions as XLA FFI custom calls (CPU).
//
// Layouts (row-major, as JAX gives them; the leading axes `[*B]` are the lanes,
// `B` is their product). A row-major `[B, n]` array is a column-major `[n, B]`
// array: the diffsol-c layout, with no copy.
//
//   diffsol_solve_dense     params [*B, np], t0 f64[], t_eval [nt]
//                                                  -> ys [*B, nt, nout]
//   diffsol_solve_fwd_sens  params [*B, np], t0 f64[], t_eval [nt]
//                                             -> ys, sens [*B, np, nt, nout]
//   diffsol_solve_vjp_fwd   params [*B, np], t0 f64[], t_eval [nt]
//                                                  -> ys, handle s64[*B]
//   diffsol_solve_vjp_bkwd  handle s64[*B], params [*B, np], t0 f64[],
//                           t_eval [nt], dgdu [*B, *D, nt, nout]
//                                                  -> grad [*B, *D, np]
//
// `nout` is the number of outputs of `out`, or `nstates` if the model has no
// `out`.
//
// `t0` is the time of the initial state: the solve starts there, and `t_eval`
// are the times it returns, all at or after it.
//
// Dtypes: `params`, `ys`, `sens`, `dgdu` and `grad` are in the model's dtype,
// and the times are always f64. diffsol-c's solve entry points take `params`,
// `t_eval` and `dgdu` as `const double *` whatever the solver's dtype is, so an
// f32 operand is widened here (`AsF64`); the tensors it returns are already in
// the model's dtype, so copying them out needs no conversion.
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "diffsol.h"
#include "registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace diffsol_jax {
namespace {

// The product of the first `n` dimensions.
template <typename Dims> size_t Product(const Dims &dims, size_t n) {
  size_t p = 1;
  for (size_t i = 0; i < n; i++)
    p *= static_cast<size_t>(dims[i]);
  return p;
}

constexpr ffi::DataType FfiType(Dtype dtype) {
  return dtype == Dtype::kF32 ? ffi::F32 : ffi::F64;
}

// Every float operand and result of a solve is in the model's dtype, except the
// times, which are always f64.
ffi::Error CheckDtype(const char *what, const ffi::AnyBuffer &buf,
                      Dtype dtype) {
  if (buf.element_type() != FfiType(dtype)) {
    return ffi::Error::InvalidArgument(std::string(what) +
                                       " must have the model's dtype, " +
                                       DtypeName(dtype));
  }
  return ffi::Error::Success();
}

// `buf` as f64, which is what diffsol-c takes whatever the solver's dtype is: a
// view of the operand when it is f64 already, else a widened copy in `scratch`.
// The caller must have checked that the dtype is one of the two.
const double *AsF64(const ffi::AnyBuffer &buf, std::vector<double> *scratch) {
  if (buf.element_type() == ffi::F64)
    return reinterpret_cast<const double *>(buf.untyped_data());
  const auto *src = reinterpret_cast<const float *>(buf.untyped_data());
  size_t n = buf.element_count();
  scratch->resize(n);
  for (size_t i = 0; i < n; i++)
    (*scratch)[i] = static_cast<double>(src[i]);
  return scratch->data();
}

// Copy a diffsol tensor into `dst` as a contiguous column-major array (which is
// the row-major array with the axes in the opposite order). Free the tensor.
//
// `dtype` is the model's, which is the dtype diffsol gives its tensors in and
// the dtype of the result `dst` points into, so the values are copied as they
// are.
ffi::Error CopyTensor(Tensor *tensor, Dtype dtype, void *dst, size_t expected) {
  const size_t width = ElementSize(dtype);
  size_t ndim = diffsol_tensor_ndim(tensor);
  size_t dims[3] = {1, 1, 1}, strides[3] = {0, 0, 0};
  for (size_t i = 0; i < ndim; i++) {
    dims[i] = diffsol_tensor_dim(tensor, i);
    strides[i] = diffsol_tensor_stride(tensor, i);
  }
  if (dims[0] * dims[1] * dims[2] != expected ||
      diffsol_tensor_dtype(tensor) != DiffsolType(dtype)) {
    diffsol_tensor_free(tensor);
    return ffi::Error::Internal(
        "diffsol returned a tensor with an unexpected shape or type");
  }
  const uint8_t *src = diffsol_tensor_ptr(tensor);
  auto *out = static_cast<uint8_t *>(dst);
  if (strides[0] == width &&
      (dims[1] == 1 || strides[1] == dims[0] * width) &&
      (dims[2] == 1 || strides[2] == dims[0] * dims[1] * width)) {
    std::memcpy(out, src, expected * width);
  } else {
    for (size_t c = 0; c < dims[2]; c++)
      for (size_t b = 0; b < dims[1]; b++)
        for (size_t a = 0; a < dims[0]; a++) {
          std::memcpy(out, src + a * strides[0] + b * strides[1] +
                               c * strides[2],
                      width);
          out += width;
        }
  }
  diffsol_tensor_free(tensor);
  return ffi::Error::Success();
}

// A locked solver for the lanes of `params`.
struct Solver {
  std::shared_ptr<Model> model;
  std::unique_lock<std::mutex> lock;
  Instance *instance = nullptr;
  size_t nbatch = 0;
  size_t nparams = 0;
  // the `params` operand as f64, and its element count
  const double *params_f64 = nullptr;
  size_t params_count = 0;
  std::vector<double> params_scratch;

  Dtype dtype() const { return model->config().dtype; }
};

// The time of the initial state, from the scalar `t0` operand. `t_eval` must
// have at least one time, and start at or after it.
ffi::ErrorOr<double> StartTime(ffi::Buffer<ffi::F64> &t0,
                               ffi::Buffer<ffi::F64> &t_eval) {
  if (t0.element_count() != 1)
    return ffi::Error::InvalidArgument("t0 must be a scalar");
  if (t_eval.element_count() == 0)
    return ffi::Error::InvalidArgument("t_eval must have at least one time");
  double start = t0.typed_data()[0];
  if (t_eval.typed_data()[0] < start) {
    return ffi::Error::InvalidArgument(
        "t_eval must start at or after t0, got t_eval[0] = " +
        std::to_string(t_eval.typed_data()[0]) +
        " and t0 = " + std::to_string(start));
  }
  return start;
}

// Set the step budget on `ode`. `diffsol_ode_get_options` hands back an owned
// handle onto the same ODE, so a set through it reaches the solver; it must be
// freed either way.
ffi::Error SetMaxSteps(Instance *instance, int64_t max_steps) {
  if (max_steps < 1)
    return ffi::Error::InvalidArgument("max_steps must be at least 1");
  OdeSolverOptions *options = nullptr;
  if (diffsol_ode_get_options(instance->ode, &options) != DIFFSOL_OK ||
      options == nullptr) {
    return ffi::Error::Internal(DiffsolError(instance));
  }
  int rc = diffsol_ode_options_set_max_number_timesteps(
      options, static_cast<size_t>(max_steps));
  diffsol_ode_options_free(options);
  if (rc != DIFFSOL_OK)
    return ffi::Error::Internal(DiffsolError(instance));
  return ffi::Error::Success();
}

ffi::Error Acquire(int64_t model_id, double rtol, double atol, double t0,
                   int64_t max_steps, size_t nbatch, Solver *s) {
  s->model = FindModel(model_id);
  if (!s->model) {
    return ffi::Error::InvalidArgument("unknown diffsol model " +
                                       std::to_string(model_id));
  }
  std::string err;
  s->instance = s->model->Acquire(nbatch, &s->lock, &err);
  if (s->instance == nullptr)
    return ffi::Error::Internal(err);
  s->nbatch = nbatch;
  s->nparams = s->model->config().nparams;
  if (diffsol_ode_set_rtol(s->instance->ode, rtol) != DIFFSOL_OK ||
      diffsol_ode_set_atol(s->instance->ode, atol) != DIFFSOL_OK ||
      diffsol_ode_set_t0(s->instance->ode, t0) != DIFFSOL_OK) {
    return ffi::Error::Internal(DiffsolError(s->instance));
  }
  if (auto e = SetMaxSteps(s->instance, max_steps); e.failure())
    return e;
  // clear the callback error of an earlier solve
  s->instance->callbacks.TakeError();
  return ffi::Error::Success();
}

// Check `params [*B, np]` against the model and return `B`.
ffi::Error Lanes(const std::shared_ptr<Model> &model,
                 const ffi::AnyBuffer &params, size_t *nbatch) {
  auto dims = params.dimensions();
  size_t np = model ? model->config().nparams : 0;
  if (dims.size() < 1 || static_cast<size_t>(dims.back()) != np) {
    return ffi::Error::InvalidArgument(
        "params must have the shape [..., nparams]");
  }
  if (model) {
    if (auto e = CheckDtype("params", params, model->config().dtype);
        e.failure()) {
      return e;
    }
  }
  *nbatch = Product(dims, dims.size() - 1);
  return ffi::Error::Success();
}

// Run a solve that returns a solution, and copy its values into `ys`.
template <typename SolveFn>
ffi::Error SolveAndCopyYs(int64_t model_id, double rtol, double atol,
                          int64_t max_steps, ffi::Buffer<ffi::F64> &t0,
                          ffi::AnyBuffer &params,
                          ffi::Buffer<ffi::F64> &t_eval, ffi::AnyBuffer &ys,
                          Solver *s, double *t0_used,
                          SolutionWrapper **solution, SolveFn solve) {
  size_t nbatch = 0;
  if (auto e = Lanes(FindModel(model_id), params, &nbatch); e.failure())
    return e;
  auto start = StartTime(t0, t_eval);
  if (!start.has_value())
    return std::move(start.error());
  *t0_used = start.value();
  if (auto e = Acquire(model_id, rtol, atol, *t0_used, max_steps, nbatch, s);
      e.failure())
    return e;
  if (auto e = CheckDtype("ys", ys, s->dtype()); e.failure())
    return e;
  s->params_f64 = AsF64(params, &s->params_scratch);
  s->params_count = params.element_count();
  size_t nt = t_eval.element_count();
  if (solve(s->instance->ode, s->params_f64, s->nparams, nbatch,
            t_eval.typed_data(), nt, solution) != DIFFSOL_OK) {
    return ffi::Error::Internal(DiffsolError(s->instance));
  }
  Tensor *tensor = nullptr;
  if (diffsol_solution_wrapper_get_ys(*solution, &tensor) != DIFFSOL_OK) {
    return ffi::Error::Internal(DiffsolError(s->instance));
  }
  return CopyTensor(tensor, s->dtype(), ys.untyped_data(),
                    s->model->nsolution() * nt * nbatch);
}

ffi::Error SolveDenseImpl(ffi::AnyBuffer params, ffi::Buffer<ffi::F64> t0,
                          ffi::Buffer<ffi::F64> t_eval,
                          ffi::Result<ffi::AnyBuffer> ys, int64_t model_id,
                          double rtol, double atol, int64_t max_steps) {
  Solver s;
  SolutionWrapper *solution = nullptr;
  double start = 0.0;
  auto e = SolveAndCopyYs(model_id, rtol, atol, max_steps, t0, params, t_eval,
                          *ys, &s, &start, &solution,
                          diffsol_ode_solve_dense);
  if (solution)
    diffsol_solution_wrapper_free(solution);
  return e;
}

ffi::Error SolveFwdSensImpl(ffi::AnyBuffer params, ffi::Buffer<ffi::F64> t0,
                            ffi::Buffer<ffi::F64> t_eval,
                            ffi::Result<ffi::AnyBuffer> ys,
                            ffi::Result<ffi::AnyBuffer> sens, int64_t model_id,
                            double rtol, double atol, int64_t max_steps) {
  Solver s;
  SolutionWrapper *solution = nullptr;
  double start = 0.0;
  auto e = SolveAndCopyYs(model_id, rtol, atol, max_steps, t0, params, t_eval,
                          *ys, &s, &start, &solution,
                          diffsol_ode_solve_fwd_sens);
  if (e.success())
    e = CheckDtype("sens", *sens, s.dtype());
  if (e.success()) {
    Tensor **list = nullptr;
    size_t len = 0;
    if (diffsol_solution_wrapper_get_sens(solution, &list, &len) !=
        DIFFSOL_OK) {
      e = ffi::Error::Internal(DiffsolError(s.instance));
    } else if (len != s.nparams) {
      e = ffi::Error::Internal(
          "diffsol returned the wrong number of sensitivities");
    } else {
      // sens j is a column-major [nout, nt, B] array; the result is row-major
      // [B, np, nt, nout]
      size_t block = s.model->nsolution() * t_eval.element_count();
      const size_t width = ElementSize(s.dtype());
      auto tmp = std::make_unique<uint8_t[]>(block * s.nbatch * width);
      auto *out = static_cast<uint8_t *>(sens->untyped_data());
      for (size_t j = 0; j < len && e.success(); j++) {
        e = CopyTensor(list[j], s.dtype(), tmp.get(), block * s.nbatch);
        list[j] = nullptr;
        for (size_t b = 0; b < s.nbatch && e.success(); b++) {
          std::memcpy(out + (b * s.nparams + j) * block * width,
                      tmp.get() + b * block * width, block * width);
        }
      }
      diffsol_tensor_list_free(list, len);
    }
  }
  if (solution)
    diffsol_solution_wrapper_free(solution);
  return e;
}

ffi::Error SolveVjpFwdImpl(ffi::AnyBuffer params, ffi::Buffer<ffi::F64> t0,
                           ffi::Buffer<ffi::F64> t_eval,
                           ffi::Result<ffi::AnyBuffer> ys,
                           ffi::ResultBuffer<ffi::S64> handle, int64_t model_id,
                           double rtol, double atol, int64_t max_steps) {
  Solver s;
  SolutionWrapper *solution = nullptr;
  VjpCheckpointWrapper *checkpoint = nullptr;
  auto solve = [&checkpoint](OdeWrapper *ode, const double *p, size_t np,
                             size_t nbatch, const double *t, size_t nt,
                             SolutionWrapper **out) {
    return diffsol_ode_solve_vjp_fwd(ode, p, np, nbatch, t, nt, out,
                                     &checkpoint);
  };
  double start = 0.0;
  auto e = SolveAndCopyYs(model_id, rtol, atol, max_steps, t0, params, t_eval,
                          *ys, &s, &start, &solution, solve);
  if (solution)
    diffsol_solution_wrapper_free(solution);
  if (e.failure()) {
    if (checkpoint)
      diffsol_vjp_checkpoint_free(checkpoint);
    return e;
  }
  CheckpointKey key;
  key.params.assign(s.params_f64, s.params_f64 + s.params_count);
  key.t_eval.assign(t_eval.typed_data(),
                    t_eval.typed_data() + t_eval.element_count());
  key.rtol = rtol;
  key.atol = atol;
  key.t0 = start;
  key.max_steps = max_steps;
  int64_t id = s.model->AddCheckpoint(s.instance, checkpoint, std::move(key));
  for (size_t b = 0; b < s.nbatch; b++)
    handle->typed_data()[b] = id;
  return e;
}

// The backward solve. It uses the checkpoint of `handle` only if the registry
// still has it, all the lanes of `handle` have the same id, and the checkpoint
// was made from exactly the same `params`, `t_eval`, `rtol`, `atol` and
// `max_steps`. If not (for example the registry freed it, or the handle comes
// from a different process), it runs the forward solve again into a temporary
// checkpoint. Thus the gradient never depends on what the registry holds.
ffi::Error SolveVjpBkwdImpl(ffi::Buffer<ffi::S64> handle, ffi::AnyBuffer params,
                            ffi::Buffer<ffi::F64> t0,
                            ffi::Buffer<ffi::F64> t_eval, ffi::AnyBuffer dgdu,
                            ffi::Result<ffi::AnyBuffer> grad, int64_t model_id,
                            double rtol, double atol, int64_t max_steps) {
  auto model = FindModel(model_id);
  if (!model)
    return ffi::Error::InvalidArgument("unknown diffsol model");
  const Dtype dtype = model->config().dtype;
  size_t nbatch = 0;
  if (auto e = Lanes(model, params, &nbatch); e.failure())
    return e;
  if (auto e = CheckDtype("dgdu", dgdu, dtype); e.failure())
    return e;
  if (auto e = CheckDtype("grad", *grad, dtype); e.failure())
    return e;
  auto start = StartTime(t0, t_eval);
  if (!start.has_value())
    return std::move(start.error());
  if (handle.element_count() != nbatch) {
    return ffi::Error::InvalidArgument(
        "handle and params have different lanes");
  }
  auto dims = dgdu.dimensions();
  size_t nout = model->nsolution();
  size_t nt = t_eval.element_count();
  if (dims.size() < 2 || static_cast<size_t>(dims.back()) != nout ||
      static_cast<size_t>(dims[dims.size() - 2]) != nt) {
    return ffi::Error::InvalidArgument(
        "dgdu must have the shape [..., nt, nout]");
  }
  size_t lanes = Product(dims, dims.size() - 2);
  if (lanes % nbatch != 0) {
    return ffi::Error::InvalidArgument(
        "the lanes of dgdu are not a multiple of the handle lanes");
  }

  // diffsol-c takes `params` and `dgdu` as f64 whatever the model's dtype is
  std::vector<double> params_scratch, dgdu_scratch;
  const double *params_f64 = AsF64(params, &params_scratch);
  const double *dgdu_f64 = AsF64(dgdu, &dgdu_scratch);

  const int64_t *ids = handle.typed_data();
  bool same_ids = true;
  for (size_t b = 1; b < nbatch; b++)
    same_ids = same_ids && ids[b] == ids[0];
  auto checkpoint = same_ids ? model->FindCheckpoint(ids[0]) : nullptr;
  if (checkpoint &&
      (checkpoint->instance->callbacks.nbatch != nbatch ||
       !checkpoint->key.Matches(params_f64, params.element_count(),
                                t_eval.typed_data(), nt, rtol, atol,
                                start.value(), max_steps))) {
    checkpoint = nullptr;
  }

  Solver s;
  VjpCheckpointWrapper *fresh = nullptr;
  const VjpCheckpointWrapper *use = nullptr;
  if (checkpoint) {
    s.model = model;
    s.instance = checkpoint->instance;
    s.lock = std::unique_lock<std::mutex>(s.instance->mutex);
    s.instance->callbacks.TakeError();
    use = checkpoint->checkpoint;
  } else {
    // recompute: the same forward solve as `diffsol_solve_vjp_fwd`
    if (auto e =
            Acquire(model_id, rtol, atol, start.value(), max_steps, nbatch, &s);
        e.failure())
      return e;
    SolutionWrapper *solution = nullptr;
    int rc = diffsol_ode_solve_vjp_fwd(s.instance->ode, params_f64, s.nparams,
                                       nbatch, t_eval.typed_data(), nt,
                                       &solution, &fresh);
    if (solution)
      diffsol_solution_wrapper_free(solution);
    if (rc != DIFFSOL_OK)
      return ffi::Error::Internal(DiffsolError(s.instance));
    model->CountRecompute();
    use = fresh;
  }
  Tensor *tensor = nullptr;
  int rc = diffsol_ode_solve_vjp_bkwd(s.instance->ode, use, dgdu_f64, nout, nt,
                                      lanes, &tensor);
  if (fresh)
    diffsol_vjp_checkpoint_free(fresh);
  if (rc != DIFFSOL_OK)
    return ffi::Error::Internal(DiffsolError(s.instance));
  return CopyTensor(tensor, dtype, grad->untyped_data(),
                    model->config().nparams * lanes);
}

} // namespace
} // namespace diffsol_jax

// `params`, `ys`, `sens`, `dgdu` and `grad` are bound as `AnyBuffer` and
// checked against the model's dtype at run time, so one handler serves both
// dtypes. The times and the checkpoint handle have fixed types.
XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveDense, diffsol_jax::SolveDenseImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveFwdSens,
                              diffsol_jax::SolveFwdSensImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveVjpFwd, diffsol_jax::SolveVjpFwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::Buffer<ffi::S64>>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveVjpBkwd,
                              diffsol_jax::SolveVjpBkwdImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("model_id")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol")
                                  .Attr<int64_t>("max_steps"));
