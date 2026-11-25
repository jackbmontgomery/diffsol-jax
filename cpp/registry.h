#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend.h"
#include "callbacks.h"
#include "diffsol.h"

namespace diffsol_jax {

struct ModelConfig {
  size_t nstates = 0;
  size_t nparams = 0;
  size_t nout = 0;
  Dtype dtype = Dtype::kF64;
  int32_t matrix_type = 0;
  int32_t linear_solver = 0;
  int32_t ode_solver = 0;
  size_t max_checkpoints = 8;
};

struct Instance {
  std::mutex mutex;
  CallbackSet callbacks;
  OdeWrapper *ode = nullptr; // made at the first solve
};

struct CheckpointKey {
  std::vector<double> params; // [B, nparams], row-major
  std::vector<double> t_eval;
  double rtol = 0.0;
  double atol = 0.0;
  double t0 = 0.0;
  int64_t max_steps = 0;
  // TODO(GPU): the device ordinal

  bool Matches(const double *p, size_t np, const double *t, size_t nt,
               double rt, double at, double start, int64_t steps) const;
};

struct Checkpoint {
  Checkpoint(int64_t id, Instance *instance, VjpCheckpointWrapper *checkpoint,
             CheckpointKey key)
      : id(id), instance(instance), checkpoint(checkpoint),
        key(std::move(key)) {}
  ~Checkpoint() { diffsol_vjp_checkpoint_free(checkpoint); }
  Checkpoint(const Checkpoint &) = delete;
  Checkpoint &operator=(const Checkpoint &) = delete;
  int64_t id;
  Instance *instance;
  VjpCheckpointWrapper *checkpoint;
  CheckpointKey key;
};

class Model {
public:
  Model(int64_t id, Backend *backend, ModelConfig config)
      : id_(id), backend_(backend), config_(config) {}
  ~Model();

  int64_t id() const { return id_; }
  const ModelConfig &config() const { return config_; }
  size_t nsolution() const {
    return config_.nout > 0 ? config_.nout : config_.nstates;
  }

  bool AddVariant(size_t nbatch, int id, size_t p, const std::string &code,
                  const std::string &options, std::string *err);
  bool HasVariant(size_t nbatch, int id, size_t p);
  bool Stats(size_t nbatch, uint64_t *ncalls, uint64_t *nstaged,
             uint64_t *nviews, bool reset);

  Instance *Acquire(size_t nbatch, std::unique_lock<std::mutex> *lock,
                    std::string *err);

  int64_t AddCheckpoint(Instance *instance, VjpCheckpointWrapper *checkpoint,
                        CheckpointKey key);
  std::shared_ptr<Checkpoint> FindCheckpoint(int64_t handle);
  void ClearCheckpoints();
  size_t NumCheckpoints();
  void CountRecompute() { nrecomputes_++; }
  uint64_t NumRecomputes() const { return nrecomputes_; }

private:
  int64_t id_;
  Backend *backend_;
  ModelConfig config_;
  std::mutex mutex_; // for `instances_` and `checkpoints_`
  std::map<size_t, std::unique_ptr<Instance>> instances_;
  std::deque<std::shared_ptr<Checkpoint>> checkpoints_;
  std::atomic<uint64_t> nrecomputes_{0};
};

// The process-wide registry.
Backend *GetBackend();
bool InitBackend(const std::string &plugin_path, std::string *err);
int64_t NewModel(const ModelConfig &config, std::string *err);
std::shared_ptr<Model> FindModel(int64_t id);
void FreeModel(int64_t id);

std::string DiffsolError(Instance *instance);

} // namespace diffsol_jax
