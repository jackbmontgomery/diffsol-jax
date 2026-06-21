#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <cstring>

namespace ffi = xla::ffi;

// ─────────────────────────────────────────────────────────────────────────────
// Rust bridge function declaration
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

int32_t diffsol_solve_rust(uint64_t handle, const double *params,
                            size_t n_params, double t0, double t_final,
                            double *ys_out, double *ts_out, size_t n_times,
                            size_t n_state, int32_t method, char *err_buf,
                            size_t err_buf_len);

} // extern "C"

// ─────────────────────────────────────────────────────────────────────────────
// XLA FFI handler — primal dense solve (the only solve entry point; derivatives
// are computed JAX-side via the augmented forward-sensitivity system).
// ─────────────────────────────────────────────────────────────────────────────

static ffi::Error SolveImpl(ffi::Buffer<ffi::F64> params,
                             ffi::Buffer<ffi::F64> t_span,
                             ffi::Result<ffi::Buffer<ffi::F64>> ys,
                             ffi::Result<ffi::Buffer<ffi::F64>> ts,
                             int64_t handle, int64_t n_times, int64_t n_state,
                             int64_t method) {
  if (t_span.dimensions().size() != 1 || t_span.dimensions()[0] != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "t_span must have shape [2]");
  }
  const double t0 = t_span.typed_data()[0];
  const double t_final = t_span.typed_data()[1];

  char err_buf[512] = {0};
  int32_t rc = diffsol_solve_rust(
      static_cast<uint64_t>(handle), params.typed_data(),
      params.dimensions()[0], t0, t_final, ys->typed_data(), ts->typed_data(),
      static_cast<size_t>(n_times), static_cast<size_t>(n_state),
      static_cast<int32_t>(method), err_buf, sizeof(err_buf));

  if (rc != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("diffsol_solve_rust: ") + err_buf);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolve, SolveImpl,
                               ffi::Ffi::Bind()
                                   .Arg<ffi::Buffer<ffi::F64>>() // params
                                   .Arg<ffi::Buffer<ffi::F64>>() // t_span
                                   .Ret<ffi::Buffer<ffi::F64>>() // ys
                                   .Ret<ffi::Buffer<ffi::F64>>() // ts
                                   .Attr<int64_t>("handle")
                                   .Attr<int64_t>("n_times")
                                   .Attr<int64_t>("n_state")
                                   .Attr<int64_t>("method"));

// ─────────────────────────────────────────────────────────────────────────────
// Handler pointer getter called from lib.rs to build the PyCapsule
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

void *get_diffsol_solve_handler() {
  return reinterpret_cast<void *>(DiffsolSolve);
}

} // extern "C"
