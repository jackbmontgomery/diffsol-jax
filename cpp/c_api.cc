#include <cstdint>
#include <cstring>
#include <string>

#include "registry.h"
#include "xla/ffi/api/c_api.h"

extern "C" XLA_FFI_Error *DiffsolSolveDense(XLA_FFI_CallFrame *);
extern "C" XLA_FFI_Error *DiffsolSolveFwdSens(XLA_FFI_CallFrame *);
extern "C" XLA_FFI_Error *DiffsolSolveVjpFwd(XLA_FFI_CallFrame *);
extern "C" XLA_FFI_Error *DiffsolSolveVjpBkwd(XLA_FFI_CallFrame *);

namespace {

int Fail(const std::string &message, char *err, size_t err_len) {
  if (err && err_len > 0) {
    std::strncpy(err, message.c_str(), err_len - 1);
    err[err_len - 1] = '\0';
  }
  return 0;
}

int Init(const char *plugin_path, char *err, size_t err_len) {
  std::string message;
  if (!diffsol_jax::InitBackend(plugin_path, &message))
    return Fail(message, err, err_len);
  return 1;
}

int64_t ModelNew(size_t nstates, size_t nparams, size_t nout, int32_t dtype,
                 int32_t matrix_type, int32_t linear_solver, int32_t ode_solver,
                 size_t max_checkpoints, char *err, size_t err_len) {
  if (!diffsol_jax::IsDtype(dtype)) {
    Fail("unknown dtype " + std::to_string(dtype), err, err_len);
    return 0;
  }
  diffsol_jax::ModelConfig config;
  config.nstates = nstates;
  config.nparams = nparams;
  config.nout = nout;
  config.dtype = static_cast<diffsol_jax::Dtype>(dtype);
  config.matrix_type = matrix_type;
  config.linear_solver = linear_solver;
  config.ode_solver = ode_solver;
  config.max_checkpoints = max_checkpoints > 0 ? max_checkpoints : 1;
  std::string message;
  int64_t id = diffsol_jax::NewModel(config, &message);
  if (id == 0)
    Fail(message, err, err_len);
  return id;
}

void ModelFree(int64_t model_id) { diffsol_jax::FreeModel(model_id); }

int ModelAddVariant(int64_t model_id, size_t nbatch, int32_t callback_id,
                    size_t p, const char *code, size_t code_len,
                    const char *options, size_t options_len, char *err,
                    size_t err_len) {
  auto model = diffsol_jax::FindModel(model_id);
  if (!model)
    return Fail("unknown model", err, err_len);
  std::string message;
  if (!model->AddVariant(nbatch, callback_id, p, std::string(code, code_len),
                         std::string(options, options_len), &message)) {
    return Fail(message, err, err_len);
  }
  return 1;
}

int ModelHasVariant(int64_t model_id, size_t nbatch, int32_t callback_id,
                    size_t p) {
  auto model = diffsol_jax::FindModel(model_id);
  return model && model->HasVariant(nbatch, callback_id, p) ? 1 : 0;
}

void ModelClearCheckpoints(int64_t model_id) {
  if (auto model = diffsol_jax::FindModel(model_id))
    model->ClearCheckpoints();
}

size_t ModelNumCheckpoints(int64_t model_id) {
  auto model = diffsol_jax::FindModel(model_id);
  return model ? model->NumCheckpoints() : 0;
}

uint64_t ModelNumRecomputes(int64_t model_id) {
  auto model = diffsol_jax::FindModel(model_id);
  return model ? model->NumRecomputes() : 0;
}

int ModelStats(int64_t model_id, size_t nbatch, uint64_t *ncalls,
               uint64_t *nstaged, uint64_t *nviews, int reset) {
  auto model = diffsol_jax::FindModel(model_id);
  return model && model->Stats(nbatch, ncalls, nstaged, nviews, reset != 0) ? 1
                                                                            : 0;
}

} // namespace

extern "C" {

struct DiffsolJaxApi {
  decltype(&Init) init;
  decltype(&ModelNew) model_new;
  decltype(&ModelFree) model_free;
  decltype(&ModelAddVariant) model_add_variant;
  decltype(&ModelHasVariant) model_has_variant;
  decltype(&ModelClearCheckpoints) model_clear_checkpoints;
  decltype(&ModelNumCheckpoints) model_num_checkpoints;
  decltype(&ModelNumRecomputes) model_num_recomputes;
  decltype(&ModelStats) model_stats;
  const void *solve_dense;
  const void *solve_fwd_sens;
  const void *solve_vjp_fwd;
  const void *solve_vjp_bkwd;
};

const DiffsolJaxApi *diffsol_jax_c_api(void) {
  static const DiffsolJaxApi api = {
      Init,
      ModelNew,
      ModelFree,
      ModelAddVariant,
      ModelHasVariant,
      ModelClearCheckpoints,
      ModelNumCheckpoints,
      ModelNumRecomputes,
      ModelStats,
      reinterpret_cast<const void *>(DiffsolSolveDense),
      reinterpret_cast<const void *>(DiffsolSolveFwdSens),
      reinterpret_cast<const void *>(DiffsolSolveVjpFwd),
      reinterpret_cast<const void *>(DiffsolSolveVjpBkwd),
  };
  return &api;
}

} // extern "C"
