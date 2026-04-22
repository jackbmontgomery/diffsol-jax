#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <string_view>

namespace ffi = xla::ffi;

extern "C" {
int32_t diffsol_solve_rust(const char *diffsl_src, size_t diffsl_src_len,
                           const double *params, size_t n_params, double t0,
                           double t_final, double *ys_out, double *ts_out,
                           size_t n_times, size_t n_state, int32_t method,
                           char *err_buf, size_t err_buf_len);

int32_t diffsol_vjp_rust(const char *diffsl_src, size_t diffsl_src_len,
                         const double *params, size_t n_params, double t0,
                         double t_final, const double *g_ys, double *grad_p_out,
                         double *grad_y0_out, size_t n_times, size_t n_state,
                         char *err_buf, size_t err_buf_len);
}

static ffi::Error SolveImpl(ffi::Buffer<ffi::F64> params,
                            ffi::Buffer<ffi::F64> t_span,
                            ffi::Result<ffi::Buffer<ffi::F64>> ys,
                            ffi::Result<ffi::Buffer<ffi::F64>> ts,
                            std::string_view diffsl_src, int64_t n_times,
                            int64_t n_state, int64_t method) {
  if (t_span.dimensions().size() != 1 || t_span.dimensions()[0] != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "t_span must have shape [2]");
  }

  const double t0 = t_span.typed_data()[0];
  const double t_final = t_span.typed_data()[1];

  char err_buf[512] = {0};
  int32_t rc = diffsol_solve_rust(
      diffsl_src.data(), diffsl_src.size(), params.typed_data(),
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
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<std::string_view>("diffsl_src")
                                  .Attr<int64_t>("n_times")
                                  .Attr<int64_t>("n_state")
                                  .Attr<int64_t>("method"));

static ffi::Error VjpImpl(ffi::Buffer<ffi::F64> params,
                          ffi::Buffer<ffi::F64> t_span,
                          ffi::Buffer<ffi::F64> g_ys,
                          ffi::Result<ffi::Buffer<ffi::F64>> grad_params,
                          ffi::Result<ffi::Buffer<ffi::F64>> grad_y0,
                          std::string_view diffsl_src, int64_t n_times,
                          int64_t n_state) {
  if (t_span.dimensions().size() != 1 || t_span.dimensions()[0] != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "t_span must have shape [2]");
  }

  const double t0 = t_span.typed_data()[0];
  const double t_final = t_span.typed_data()[1];

  char err_buf[512] = {0};
  int32_t rc =
      diffsol_vjp_rust(diffsl_src.data(), diffsl_src.size(),
                       params.typed_data(), params.dimensions()[0], t0, t_final,
                       g_ys.typed_data(), grad_params->typed_data(),
                       grad_y0->typed_data(), static_cast<size_t>(n_times),
                       static_cast<size_t>(n_state), err_buf, sizeof(err_buf));

  if (rc != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("diffsol_vjp_rust: ") + err_buf);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolVjp, VjpImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Attr<std::string_view>("diffsl_src")
                                  .Attr<int64_t>("n_times")
                                  .Attr<int64_t>("n_state"));
