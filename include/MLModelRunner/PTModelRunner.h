#ifndef PTMODELRUNNER_H
#define PTMODELRUNNER_H

#include "MLModelRunner/MLModelRunner.h"
#include "SerDes/TensorSpec.h"
#include "SerDes/pytorchSerDes.h"
#include "llvm/Support/ErrorHandling.h"
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h> // or model_container_runner_cuda.h for CUDA

#include <memory>
#include <vector>


namespace MLBridge {

/// PTModelRunner - PyTorch Compiled model implementation of the
/// MLModelRunner. It uses an AOT-compiled model for efficient execution.
template <class TGen> class PTModelRunner final : public MLModelRunner {
public:
  // New constructor that takes the model path as an input
  PTModelRunner(const std::string &modelPath, llvm::LLVMContext &Ctx)
      : MLModelRunner(MLModelRunner::Kind::PTAOT, BaseSerDes::Kind::Pytorch, &Ctx),
        CompiledModel(std::make_unique<PytorchSerDes>(modelPath)) {

    SerDes->setRequest(CompiledModel.get());

    assert(CompiledModel && "The CompiledModel should be valid");
  }

  virtual ~PTModelRunner() = default;

  virtual void requestExit() override {
    llvm_unreachable("requestExit() is not supported in PTModelRunner");
  }

  static bool classof(const MLModelRunner *R) {
    return R->getKind() == MLModelRunner::Kind::PTAOT;
  }

private:
  void *evaluateUntyped() override;

  std::unique_ptr<PytorchSerDes> CompiledModel;
};

/// A mock class satisfying the interface expected by PTModelRunner for
/// its `TGen` parameter. Useful to avoid conditional compilation complexity, as
/// a compile-time replacement for a real AOT-ed model.
class NoopSavedModelImpl final {
#define NOOP_MODEL_ERRMSG                                                      \
  "The mock AOT-ed saved model is a compile-time stub and should not be "      \
  "called."

public:
  NoopSavedModelImpl() = default;
  int LookupArgIndex(const std::string &) {
    llvm_unreachable(NOOP_MODEL_ERRMSG);
  }
  int LookupResultIndex(const std::string &) {
    llvm_unreachable(NOOP_MODEL_ERRMSG);
  }
  void Run() { llvm_unreachable(NOOP_MODEL_ERRMSG); }
  void *result_data(int) { llvm_unreachable(NOOP_MODEL_ERRMSG); }
  void *arg_data(int) { llvm_unreachable(NOOP_MODEL_ERRMSG); }
#undef NOOP_MODEL_ERRMSG
};

template <class T> bool isEmbeddedModelEvaluatorValid() { return true; }

template <> inline bool isEmbeddedModelEvaluatorValid<NoopSavedModelImpl>() {
  return false;
}
} // namespace MLBridge

#endif // TFMODELRUNNER_H
