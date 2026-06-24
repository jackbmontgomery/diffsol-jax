#include "xla/ffi/api/ffi.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace ffi = xla::ffi;

extern "C" {

int32_t diffsol_solve_f64(uint64_t handle, const double *params,
                          size_t n_params, const double *t_eval, size_t n_times,
                          int32_t method, double h0, double rtol, double atol,
                          double *ys_out, double *ts_out, char *err_buf,
                          size_t err_buf_len);

int32_t diffsol_solve_f32(uint64_t handle, const double *params,
                          size_t n_params, const double *t_eval, size_t n_times,
                          int32_t method, double h0, double rtol, double atol,
                          float *ys_out, float *ts_out, char *err_buf,
                          size_t err_buf_len);
}

static ffi::Error SolveImplF64(ffi::Buffer<ffi::F64> params,
                               ffi::Buffer<ffi::F64> t_eval,
                               ffi::Result<ffi::Buffer<ffi::F64>> ys,
                               ffi::Result<ffi::Buffer<ffi::F64>> ts,
                               int64_t handle, int64_t method, double h0,
                               double rtol, double atol) {

  char err_buf[512] = {0};
  int32_t rc = diffsol_solve_f64(
      static_cast<uint64_t>(handle), params.typed_data(),
      params.dimensions()[0], t_eval.typed_data(), t_eval.dimensions()[0],
      static_cast<int32_t>(method), static_cast<double>(h0),
      static_cast<double>(rtol), static_cast<double>(atol), ys->typed_data(),
      ts->typed_data(), err_buf, sizeof(err_buf));

  if (rc != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("diffsol_solve_rust_f64: ") + err_buf);
  }
  return ffi::Error::Success();
}

static ffi::Error SolveImplF32(ffi::Buffer<ffi::F32> params,
                               ffi::Buffer<ffi::F32> t_eval,
                               ffi::Result<ffi::Buffer<ffi::F32>> ys,
                               ffi::Result<ffi::Buffer<ffi::F32>> ts,
                               int64_t handle, int64_t method, double h0,
                               double rtol, double atol) {
  const size_t n_params = params.dimensions()[0];
  const size_t n_times = t_eval.dimensions()[0];

  // diffsol always reads inputs as f64, so widen f32 -> f64 here
  const float *params_f32 = params.typed_data();
  const float *t_eval_f32 = t_eval.typed_data();
  std::vector<double> params_f64(params_f32, params_f32 + n_params);
  std::vector<double> t_eval_f64(t_eval_f32, t_eval_f32 + n_times);

  char err_buf[512] = {0};
  int32_t rc = diffsol_solve_f32(
      static_cast<uint64_t>(handle), params_f64.data(), n_params,
      t_eval_f64.data(), n_times, static_cast<int32_t>(method),
      static_cast<double>(h0), static_cast<double>(rtol),
      static_cast<double>(atol), ys->typed_data(), ts->typed_data(), err_buf,
      sizeof(err_buf));

  if (rc != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("diffsol_solve_rust_f32: ") + err_buf);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveF64, SolveImplF64,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>() // params
                                  .Arg<ffi::Buffer<ffi::F64>>() // t_eval
                                  .Ret<ffi::Buffer<ffi::F64>>() // ys
                                  .Ret<ffi::Buffer<ffi::F64>>() // ts
                                  .Attr<int64_t>("handle")
                                  .Attr<int64_t>("method")
                                  .Attr<double>("h0")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(DiffsolSolveF32, SolveImplF32,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>() // params
                                  .Arg<ffi::Buffer<ffi::F32>>() // t_eval
                                  .Ret<ffi::Buffer<ffi::F32>>() // ys
                                  .Ret<ffi::Buffer<ffi::F32>>() // ts
                                  .Attr<int64_t>("handle")
                                  .Attr<int64_t>("method")
                                  .Attr<double>("h0")
                                  .Attr<double>("rtol")
                                  .Attr<double>("atol"));

extern "C" {
void *get_diffsol_solve_handler_f64() {
  return reinterpret_cast<void *>(DiffsolSolveF64);
}

void *get_diffsol_solve_handler_f32() {
  return reinterpret_cast<void *>(DiffsolSolveF32);
}
}
