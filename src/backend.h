#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "dtype.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace diffsol_jax {

// The maximum number of arguments of a callback executable: (v, p, t, beta, y)
// for `mass`.
constexpr int kMaxArgs = 5;
// The maximum number of axes of an argument.
constexpr int kMaxDims = 3;

// A view of caller memory: a dense row-major array of `dtype` values.
struct ArgView {
  const void *ptr;
  int64_t dims[kMaxDims];
  int ndim;
  Dtype dtype;
};

// One PJRT client (one process, one platform), and its first device.
class Backend {
public:
  static Backend *Create(const std::string &plugin_path, std::string *err);
  PJRT_LoadedExecutable *Compile(const std::string &code,
                                 const std::string &options, std::string *err);

  void Destroy(PJRT_LoadedExecutable *exe);

  PJRT_Buffer *MakeView(const ArgView &arg, std::string *err);
  void DestroyBuffer(PJRT_Buffer *buffer);

  // TODO(GPU): a device plugin's requirement, when there is one.
  size_t view_alignment() const { return 64; }

  bool Execute(PJRT_LoadedExecutable *exe, PJRT_Buffer *const *args, int nargs,
               void *out, size_t out_bytes, std::string *err);

  const PJRT_Api *api() const { return api_; }

private:
  Backend() = default;
  bool Check(PJRT_Error *error, const char *what, std::string *err);
  void AwaitAndDestroy(PJRT_Event *event, std::string *err, bool *ok);
  // Get the `out_bytes` of a finished execution's output buffer into `out`.
  bool CopyOut(PJRT_Buffer *output, void *out, size_t out_bytes,
               std::string *err);

  const PJRT_Api *api_ = nullptr;
  PJRT_Client *client_ = nullptr;
  PJRT_Device *device_ = nullptr;
};

} // namespace diffsol_jax
