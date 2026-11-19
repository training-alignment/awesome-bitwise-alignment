# Awesome Bitwise Alignment [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

**English** · [简体中文](#中文简介)

> A curated list of work on **bitwise reproducibility, determinism, batch invariance, and train–inference alignment** in ML systems, mostly LLMs.

Two runs that should compute the same thing often don't. The cause is a different batch size, a different reduction order, a different kernel, a different parallel layout, a library upgrade, or a faulty GPU. Floating-point addition is not associative, so each of these can change the bits. In pre-training, that makes a regression impossible to separate from noise. In RL, the rollout engine and the trainer disagree on log-probabilities, so "on-policy" training is quietly off-policy. In serving, `temperature=0` stops being reproducible.

This list collects how labs and frameworks make computation exact, what exactness costs, how to correct for mismatch statistically when exactness is too expensive, and the tools that find where two runs first diverge.

<!-- site:start -->
<!-- generated from data/timeline.json by scripts/build.py; edit the JSON, not this table -->

▶ **[Interactive history](https://training-alignment.github.io/awesome-bitwise-alignment/)** (English / 简体中文 / bilingual side by side): a short scroll-through story: why bits differ, the key milestones, and what exactness costs. Every event, with its source, is on the explore page.

<!-- site:end -->

> **Coming soon: the Training Alignment Handbook.** Recipes, checklists and case studies for making training stacks agree bit for bit, and for finding out why when they don't.

## Bitwise recipes that ship today

<!-- recipes:start -->
<!-- generated from data/timeline.json by scripts/build.py; edit the JSON, not this table -->

What each stack turns on to get exact bits, with the cost or limit its own docs state. Logos identify the organization behind each recipe.

| | Who | Turn on | Covers | Stated cost / limit |
|:-:|---|---|---|---|
| <img src="https://cdn.simpleicons.org/google" width="22" height="22" alt="Google"> | **Google** · [JAX / XLA](https://openxla.org/xla/determinism) | `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0` | Training | Slower scatter; no live autotuning |
| <img src="https://cdn.simpleicons.org/pytorch" width="22" height="22" alt="PyTorch"> | [**PyTorch**](https://docs.pytorch.org/docs/stable/notes/randomness.html) | `torch.use_deterministic_algorithms(True) · CUBLAS_WORKSPACE_CONFIG=:4096:8` | Training | Same hardware and release only |
| <img src="https://cdn.simpleicons.org/pytorch" width="22" height="22" alt="PyTorch"> | **PyTorch** · [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/rl/BITWISE_PARITY.md) | `DebugConfig(batch_invariant=True, deterministic=True)` | RL trainer ↔ vLLM | Trainer and generator need the same TP degree |
| <img src="https://cdn.simpleicons.org/nvidia" width="22" height="22" alt="NVIDIA"> | **NVIDIA** · [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/deterministic-training.html) | `--deterministic-mode` | Pre-training | ~1.5% at 3,072 GPUs |
| <img src="https://cdn.simpleicons.org/nvidia" width="22" height="22" alt="NVIDIA"> | **NVIDIA** · [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#experimental-batch-invariance) | `CUBLAS_BATCH_INVARIANCE_FLAGS` | GEMM | Split-K off; “may reduce performance” |
|  | **Thinking Machines Lab** · [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) | `with set_batch_invariant_mode():` | Inference kernels | Matmul ~20% slower than cuBLAS |
| <img src="https://cdn.simpleicons.org/vllm" width="22" height="22" alt="vLLM"> | [**vLLM**](https://docs.vllm.ai/en/latest/features/batch_invariance/) | `VLLM_BATCH_INVARIANT=1` | Inference | “May reduce performance” |
|  | **SGLang (LMSYS)** · [SGLang](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) | `--enable-deterministic-inference` | Inference | 34.35% average slowdown |
| <img src="https://cdn.simpleicons.org/bytedance" width="22" height="22" alt="ByteDance Seed"> | **ByteDance Seed** · [verl](https://verl.readthedocs.io/en/latest/advance/determinism.html) | `fsdp_config.batch_invariant=true` | RL | Slower steps on SM80 (Triton GEMMs) |
|  | **SkyRL** · [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) | one execution contract for vLLM + Megatron | RL | +25.3% per RL step |
| <img src="https://cdn.simpleicons.org/amd" width="22" height="22" alt="AMD"> | **AMD** · [Primus](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/determinism-and-reproducibility.html) | `PRIMUS_DETERMINISTIC=1` | Training on ROCm | “For debugging and validation” |
| <img src="https://cdn.simpleicons.org/huggingface" width="22" height="22" alt="Hugging Face"> | **Hugging Face** · [Transformers](https://huggingface.co/docs/transformers/main/en/internal/trainer_utils) | `enable_full_determinism(seed)` | Training | set_seed(deterministic=True) “can slow down training” |
| <img src="https://cdn.simpleicons.org/tensorflow" width="22" height="22" alt="TensorFlow"> | [**TensorFlow**](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism) | `tf.config.experimental.enable_op_determinism()` | Training | tf.data can be much slower |

<!-- recipes:end -->

**Entry format.** `Title — Authors/Org · Date · What it does. Cost/limits as stated by the source.` Every link was opened and every number appears in the linked source, as of 2026-11-18. A dated "living doc" is a documentation page with no publication date. Text marked *our reading* is the curator's interpretation, not the source's.

## Contents

- [Bitwise recipes that ship today](#bitwise-recipes-that-ship-today)
- [Start here](#start-here)
- [How labs and frameworks implement bitwise reproducibility](#how-labs-and-frameworks-implement-bitwise-reproducibility)
  - [Vendor libraries: cuBLAS, cuDNN, NCCL, Transformer Engine](#vendor-libraries-cublas-cudnn-nccl-transformer-engine)
  - [Batch-invariant kernels](#batch-invariant-kernels)
  - [Training frameworks](#training-frameworks)
  - [Inference engines](#inference-engines)
  - [Compilers and other frameworks: XLA, TensorFlow, Hugging Face](#compilers-and-other-frameworks-xla-tensorflow-hugging-face)
  - [Hosted APIs and providers](#hosted-apis-and-providers)
- [By use case](#by-use-case)
  - [Pre-training reproducibility and debugging](#pre-training-reproducibility-and-debugging)
  - [RL train–inference alignment (sync and async)](#rl-traininference-alignment-sync-and-async)
  - [Inference determinism](#inference-determinism)
  - [Framework migration, upgrade equivalence, and CI](#framework-migration-upgrade-equivalence-and-ci)
  - [Hardware faults and silent data corruption (SDC)](#hardware-faults-and-silent-data-corruption-sdc)
- [Tools and debuggers](#tools-and-debuggers)
- [What exactness costs](#what-exactness-costs)
- [Timeline](#timeline)
- [Not yet verified](#not-yet-verified)
- [Contributing](#contributing) · [License](#license) · [中文简介](#中文简介)

---

## Start here

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) — Horace He, Thinking Machines Lab · 2025-09-10 · The essay that reframed the field. Missing **batch invariance**, not "concurrency + floating point", is the main source of inference nondeterminism. Introduces batch-invariant RMSNorm, matmul and attention (with a fixed split *size*). 1,000 temp-0 Qwen3-235B completions went from 80 unique outputs to 1. Cost on Qwen3-8B in vLLM: 26 s default, 55 s deterministic, 42 s with improved attention. A bitwise-identical sampler and trainer gives KL = 0 in RL.
- [PyTorch: Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) — PyTorch · living doc · Batched results are not guaranteed to match sliced ones (`(A@B)[0]` vs `A[0]@B[0]`). Also covers TF32 defaults and `allow_bf16_reduced_precision_reduction`. Nothing is bitwise across releases.
- [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning](https://arxiv.org/abs/2408.05148) — Shanmugavelu et al. · 2024-08 · Measures the cost of replacing GPU atomics with deterministic alternatives. Finds PyTorch's determinism docs incomplete. Groq hardware is deterministic by design.
- [Problems and Opportunities in Training Deep Learning Software Systems: An Analysis of Variance](https://hvpham.github.io/publication/2020-ASE-Variance) — Pham, Qian, Wang, Lutellier, Rosenthal, Tan, Yu, Nagappan · ASE 2020 · Identical training runs differ by up to 10.8% accuracy, 2.9% from implementation-level nondeterminism alone. 83.8% of 901 surveyed practitioners were unaware or unsure.
- [ML Engineering: Reproducibility](https://github.com/stas00/ml-engineering/blob/master/training/reproducibility/README.md) — Stas Bekman · living book · Practical checklist: seeds, deterministic cuDNN/cuBLAS, and logging the environment.

## How labs and frameworks implement bitwise reproducibility

### Vendor libraries: cuBLAS, cuDNN, NCCL, Transformer Engine

- [cuBLAS: Results reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) — NVIDIA · living doc · Bitwise-identical every run only on the same architecture, SM count and toolkit version. Breaks with multiple streams; fix with a workspace per stream or `CUBLAS_WORKSPACE_CONFIG=:16:8` (may cost performance) or `:4096:8` (about 24 MiB more memory). Atomics mode is not reproducible.
- [cuBLAS: Experimental batch invariance](https://docs.nvidia.com/cuda/cublas/index.html#experimental-batch-invariance) — NVIDIA · living doc (cuBLAS 13.4.2+) · `CUBLASLT_MATMUL_DESC_BATCH_INVARIANCE_FLAGS` or `CUBLAS_BATCH_INVARIANCE_FLAGS`: any M/N sub-range matches the full problem bitwise. Blackwell (CC 10.x) only. Split-K disabled and algorithm choice restricted, "which may reduce performance".
- [cuDNN: Reproducibility (determinism)](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.2.1/developer/misc.html) — NVIDIA · living doc (v9.2.1) · Most routines are bitwise reproducible on one architecture. Atomics-based exceptions include some ConvBwdFilter/ConvBwdData algorithms, max-pool backward and CTC loss. No routine is guaranteed across architectures.
- [NVIDIA/nccl #1055: Ring vs Tree precision](https://github.com/NVIDIA/nccl/issues/1055) — NCCL maintainers · 2023-11-06 · Summation order differs by algorithm and channel, so results change with `NCCL_ALGO` and the channel count. There is no official NCCL determinism doc.
- [Transformer Engine: environment variables](https://nvidia.github.io/TransformerEngine/envvars.html) — NVIDIA · living doc · `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` restricts TE (PyTorch and JAX) to deterministic algorithms.
- [FlashAttention (`deterministic=True` backward)](https://github.com/Dao-AILab/flash-attention) — Tri Dao / Dao-AILab · living doc · Forward is always deterministic. The deterministic backward is "slightly slower and uses more memory".
- [NVIDIA/framework-reproducibility](https://github.com/NVIDIA/framework-reproducibility) — NVIDIA · 2019– (renamed 2023-02-14) · Formerly `tensorflow-determinism` / `framework-determinism`. Status tables, patches and a Seeder tool for GPU determinism in TF and PyTorch.

### Batch-invariant kernels

- [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — Thinking Machines Lab · 2025-09 · Swaps in batch-invariant Triton `mm`, `addmm`, `log_softmax` and `mean` through `torch.Library` under `set_batch_invariant_mode()`. A vLLM proof of concept went from 18 unique outputs to 1.
- [Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch (TBIK)](https://arxiv.org/abs/2511.17826) — Zhang, Ding, Yuan, Liu, Mao, Xing, Liu · 2025-11-21 (ICML 2026) · Tree-based invariant Triton kernels give bitwise-identical results across TP sizes and between vLLM and FSDP.
- [DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training](https://arxiv.org/abs/2601.21824) — Qiang et al. (SJTU) · 2026-01-29 (ICLR 2026) · FA3's deterministic backward loses up to 37.9% throughput. DASH's schedules recover up to 1.28× on H800.
- [Accelerating the Mitigation of LLM Inference Nondeterminism Across GPU Architectures](https://arxiv.org/abs/2609.25624) — Cooper, Jeong, Jeon, Young, Kim · 2026-09-22 · Fused-upcast GEMMs with reduction order fixed by shape give bitwise-identical linear layers across Ampere, Ada and Hopper. 1.17–3.1× faster end to end than prior work.
- [RepDL: Bit-level Reproducible Deep Learning Training and Inference](https://arxiv.org/abs/2510.09180) — Xie, Zhang, Chen (Microsoft) · 2025-10-10 · Correct rounding plus order invariance for bit-level reproducibility across environments.

### Training frameworks

- [PyTorch: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — PyTorch · living doc · Covers `torch.use_deterministic_algorithms()`, `cudnn.deterministic` / `benchmark=False`, and `fill_uninitialized_memory`. Flash/efficient SDPA backward is nondeterministic. No guarantee across releases or platforms.
- [Megatron Core: Deterministic training](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/deterministic-training.html) — NVIDIA · living doc · `--deterministic-mode` restricts `NCCL_ALGO` to Ring/CollNet (no Tree), sets `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and `CUBLAS_WORKSPACE_CONFIG`, and pins Triton autotune. A bit-exact suite covers TP/PP/VPP/EP/FSDP and FP8/FP4. Rejects `--tp-comm-overlap`. The uninitialized-memory fill costs about 15% TFLOP/s, so it is turned off.
- [Scale Bitwise-Deterministic Pretraining with NVIDIA Megatron Core](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) — Aithal, Harper, Bhavani, Xu, Li (NVIDIA) · 2026-10-06 · Per-rank `torch.hash_tensor` fingerprint traces localize divergence to a kernel. Also fixes a grouped-GEMM epilogue race. The determinism tax on Nemotron 3 Ultra fell from about 17% to 1.5% at 3,072 GPUs. Same hardware and software only.
- [Megatron-LM #8708: batch-invariant GEMMs on cuBLASLt with a starved workspace](https://github.com/NVIDIA/Megatron-LM/pull/8708) — Lawrence McAfee (NVIDIA) · 2026-10-29 (open) · Fixes split-K leakage in `set_batch_invariant_mode` for RL logprob parity. Training GEMMs may be slower.
- [TorchTitan: debugging (deterministic mode, loss compare)](https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md) — PyTorch · living doc · `debug.deterministic=True`; `scripts/loss_compare.py` checks loss bit-identically. "Significantly reduce training performance". Do not validate with `deterministic_warn_only`.
- [TorchTitan RL: BITWISE_PARITY.md](https://github.com/pytorch/torchtitan/blob/main/torchtitan/rl/BITWISE_PARITY.md) — PyTorch · 2026-10 · Recipe for bitwise trainer/vLLM parity: batch-invariant ops, FA3 `num_splits=1`, a fixed NCCL tree with one channel, TF32 off, and an fp32 RoPE cache. Trainer and generator need the same TP degree; no sequence parallelism.
- [AMD Primus: Determinism and reproducibility](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/determinism-and-reproducibility.html) — AMD · living doc · `PRIMUS_DETERMINISTIC=1` sets Ring collectives and `ROCBLAS_DEFAULT_ATOMICS_MODE=0`, and turns off torch.compile and autotune, for Megatron, TorchTitan and MaxText. "For debugging and validation, not production throughput."

### Inference engines

- [vLLM: Batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) — vLLM · living doc (beta) · `VLLM_BATCH_INVARIANT=1` on NVIDIA SM ≥ 8.0, AMD Instinct and Intel XPU. TP all-reduces that are not a multiple of 16 B are unsupported. Disables FlashInfer, AITER and QuickReduce all-reduce; "may reduce performance".
- [vLLM: Reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/) — vLLM · living doc · Not reproducible by default. Reproducible only on the same hardware and version. Offline needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` or batch invariance; online needs batch invariance.
- [vLLM #25603: Kernel-override determinism [1/n]](https://github.com/vllm-project/vllm/pull/25603) — Bram Wasti · 2025-09-26 (merged) · The first batch-invariance plumbing in vLLM.
- [vLLM #27433: Batch-invariant feature and performance optimization](https://github.com/vllm-project/vllm/issues/27433) — Wentao Ye · 2025-10-23 · Tracking issue for backends (FlashInfer, MLA, DeepGEMM), CUDA graphs, FP8/NVFP4, ROCm and spec decode.
- [Towards Deterministic Inference in SGLang and Reproducible RL Training](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) — SGLang Team (LMSYS) · 2025-09-22 · `--enable-deterministic-inference` works with chunked prefill, CUDA graphs and radix cache. Average slowdown 34.35% (FlashInfer/FA3) vs 61.5% for the TML baseline. Dense models, TP ≤ 2. Recommended "primarily for debugging and reproducibility".
- [SGLang docs: Deterministic inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference) — SGLang · living doc · Backend compatibility table, default `sampling_seed` 42, and `python3 -m sglang.test.test_deterministic`.
- [SGLang #10278: Deterministic inference roadmap](https://github.com/sgl-project/sglang/issues/10278) — SGLang · 2025-09-10 · Deterministic all-reduce for large TP, deterministic MoE, DeepSeek-V3 and FP8. Still open at closure: Blackwell TP4, EP, spec decoding.
- [NVIDIA NIM: Deterministic generation mode](https://archive.docs.nvidia.com/nim/large-language-models/1.10.0/deterministic-mode.html) — NVIDIA · living doc (NIM 1.10) · `NIM_FORCE_DETERMINISTIC=1` on TensorRT-LLM buildable profiles only; "may slightly reduce" throughput.

### Compilers and other frameworks: XLA, TensorFlow, Hugging Face

- [OpenXLA: Determinism](https://openxla.org/xla/determinism) — OpenXLA · living doc · Two sources: compile-time autotuning and runtime atomics. `--xla_gpu_exclude_nondeterministic_ops` and `--xla_gpu_autotune_level=0`. Slower scatter; compile error when an op has no deterministic implementation.
- [openxla/xla #12520](https://github.com/openxla/xla/pull/12520) — Ilia Sergachev · 2024-05-17 · Adds `xla_gpu_exclude_nondeterministic_ops`. The older `xla_gpu_deterministic_ops` also disables autotuning.
- [tf.config.experimental.enable_op_determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism) — Google · living doc (TF 2.8+) · Deterministic ops on fixed hardware and software. Unimplemented ops raise an error. `tf.data` can become "several orders of magnitude slower".
- [Hugging Face Transformers: `enable_full_determinism`](https://huggingface.co/docs/transformers/main/en/internal/trainer_utils) — Hugging Face · living doc · Sets `CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC=1` and `HCCL_DETERMINISTIC=1`, turns on deterministic algorithms, and turns off cuDNN benchmark. "Can slow down training."

### Hosted APIs and providers

- [OpenAI Cookbook: Reproducible outputs with the seed parameter](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter) — OpenAI · 2023-11 · `seed` plus `system_fingerprint` give "best effort" determinism. Outputs are "mostly identical" when both match; not guaranteed.
- [Vertex AI: Content generation parameters (seed)](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/content-generation-parameters) — Google Cloud · living doc · Seed is best effort: "Deterministic output isn't guaranteed."
- [Fireworks: Frontier RL training infrastructure for GLM 5.2](https://fireworks.ai/blog/frontier-rl-training-infrastructure-glm-5-2) — Fireworks AI · 2026-06-24 · Managed batch-invariant serving with zero-KLD train/serve alignment; temp-0 outputs don't change with load. Claims "virtually none" of the open-source deterministic-mode tax. Vendor claim, no methodology.

## By use case

### Pre-training reproducibility and debugging

**Determinism studies**

- [Deterministic Implementations for Reproducibility in Deep Reinforcement Learning](https://arxiv.org/abs/1809.05676) — Nagarajan, Warnell, Stone · 2018-09 · Builds a fully deterministic DQN, then adds back one source of nondeterminism at a time (including GPU ops). Each source alone can change results noticeably.
- [Towards Training Reproducible Deep Learning Models](https://arxiv.org/abs/2202.02326) — Chen, Wen, Shi, Lin, Rajbahadur, Jiang · ICSE 2022 · Record-and-replay for software randomness and profile-and-patch for hardware nondeterminism. Reproduced 6 open-source models and 1 commercial model.
- [PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs](https://arxiv.org/abs/2503.09543) — van der Wal et al. · ICLR 2025 · 45 seed-varied Pythia runs and about 7k checkpoints. Training dynamics are consistent across seeds, so outlier runs are detectable.

**Bitwise-deterministic pretraining at scale**

- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al. (Google) · 2022-04 · §5: "The model is fully bitwise reproducible from any checkpoint." Achieved with a bitwise-deterministic JAX + XLA + T5X stack and a deterministic data pipeline, where each batch is a function of the step number only. The data-ordering method is patented: [US12014276](https://patents.google.com/patent/US12014276) (Google, priority date 2022-04-01).
- [Scale Bitwise-Deterministic Pretraining with NVIDIA Megatron Core](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) — NVIDIA · 2026-10-06 · See [Training frameworks](#training-frameworks). The determinism tax is about 1.5% at 3,072 GPUs.
- [OPEN-1B: A Fully Auditable Training Run](https://arxiv.org/abs/2609.17380) — Donaghy, Wilcox, Ersoy et al. · 2026-09 · Bitwise-reproducible training on heterogeneous commodity hardware with fixed reduction, data and collective order. Each step can be verified by independent auditors.
- [Reproducibility is the New Copyleft: Defining AGI-oriented Reproducible Builds](https://arxiv.org/abs/2606.03019) — Masayuki Hatta · 2026-06 (AGI-26) · Policy paper arguing for bit-exact reproducible model builds; defines 7 requirements.

**Reproducing open runs across stacks**

- [Olmo3 reproduction in TorchTitan](https://docs.pytorch.org/devlogs/distributed/2026-09-02-olmo3-reproduction/) — Ruisi Zhang (PyTorch) · 2026-09-02 · Reproduced 23% of the Olmo3-7B loss curve on 512 H100s. Aligning RoPE, RMSNorm, fused QKV and the grad-norm reduction order was needed to approach bitwise. Single run, no error bars.
- [Reproducing Olmo 3 7B Pre-training in MaxText](https://developers.googleblog.com/en/reproducing-olmo-3-7b-pre-training-in-maxtext-case-study-of-large-scale-training-on-tpus/) — Google with Ai2 · 2026-09 · Matches within run-to-run noise, not bitwise. Found a data-loader double-sharding bug: training loss looked 0.22 better while held-out evals did not move.

**Open logbooks and reports**

- [OPT-175B chronicles](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) — Meta AI · 2022 · The full training logbook of OPT-175B.
- [BLOOM tr11-176B chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) — BigScience · 2022 · Records a BF16 optimizer bug that skipped grad clipping for TP > 1 and desynced layer norms, plus loss spikes and GPU failures.
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) — Biderman et al. (EleutherAI) · ICML 2023 · Identical data order across 16 models, plus tools to rebuild the exact dataloader.
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) — Groeneveld et al. (Ai2) · 2024-02 · Releases data and training code with the weights.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Meta · 2024-07 · Of 419 unexpected interruptions in 54 days, 6 came from SDC. Comparing loss across parallelism setups exposed numerical bugs. Uses FP32 gradient accumulation and reduce-scatter.

**Instabilities and numerics**

- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) — Wortsman et al. (Google DeepMind) · 2023-09 · Reproduces logit-growth instabilities at small scale. AdamW's default epsilon is too large at scale.
- [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) — Molybog et al. (Meta) · 2023-04 · Loss spikes at 7B–546B are attributed to Adam updates becoming uncorrelated with the descent direction.
- [Is Flash Attention Stable?](https://arxiv.org/abs/2405.02803) — Golden et al. (Meta FAIR, Harvard) · 2024-05 · FlashAttention shows about 10× more forward numeric deviation than baseline attention in BF16, but 2–5× less weight deviation than low-precision training.
- [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) — Fishman, Chmiel, Banner, Soudry · ICLR 2025 · FP8 instability appears only in long runs. Traced to SwiGLU outlier amplification and fixed with Smooth-SwiGLU.
- [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation) — Hugging Face · 2024-10 · Per-micro-batch loss averaging made gradient accumulation differ from the full batch. Fixed via `num_items_in_batch`.

**Silent training bugs**

- [Towards Understanding Bugs in Distributed Training and Inference Frameworks for Large Language Models](https://arxiv.org/abs/2506.10426) — Yu, Chen, Niu, Hu, Keung, Xia · 2025-06 · 308 bugs in DeepSpeed, Megatron-LM and Colossal-AI. Symptoms are often far from the root cause; about 48% of fixes are 10 lines or fewer.
- [A Comprehensive Study of Bugs in Modern Distributed Deep Learning Systems](https://arxiv.org/abs/2512.20345) — Ma et al. · 2025-12 · 849 issues: 34 symptoms, 28 root causes, 6 fix patterns. 45.1% of symptoms occur only in distributed settings.
- [L4: Diagnosing Large-scale LLM Training Failures via Automated Log Analysis](https://arxiv.org/abs/2503.20263) — Jiang et al. (CUHK, Huawei) · FSE 2025 · 428 production failures; localizes them from logs.
- See also [TrainCheck](#tools-and-debuggers), [TTrace](#tools-and-debuggers) and [OpGuard](#tools-and-debuggers).

### RL train–inference alignment (sync and async)

The rollout engine (vLLM, SGLang) and the trainer (FSDP, Megatron) run the same weights with different kernels, so their token probabilities differ. That makes "on-policy" RL off-policy and can collapse training. There are two responses: make the bits identical, or correct for the gap statistically. Async RL adds a second, separate gap, **policy staleness**. Several sources below keep the two apart. Bin 0 in prime-rl's staleness metric "holds only the numerical mismatch".

**Bitwise / exact alignment systems**

- [No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) — Wasti, Ye, Rao, Goin, Zhang, Liu, Gimelshein, Kwon, You, Li · 2025-11-10 · Uses vLLM batch-invariant forward kernels with hand-written backward passes (Qwen3-1.7B, GSM8K). "The bitwise RL run is 2.4x slower". Eager only, with duplicated model code.
- [Introducing Miles](https://www.lmsys.org/blog/2025-11-19-miles) — RadixArk · 2025-11-19 · "True on-policy" with zero KL from FA3, DeepGEMM, TML batch-invariant kernels and torch.compile.
- [Miles: True on-policy (docs)](https://github.com/radixark/miles/blob/main/docs/examples/infra-features/true-on-policy.md) — RadixArk · living doc · `train_rollout_logprob_abs_diff` is exactly 0 for Qwen3-4B (SGLang ↔ FSDP). Rollout has "an acceptable slowdown" (no figure).
- [slime: Reproducibility](https://github.com/THUDM/slime/blob/main/docs/en/advanced/reproducibility.md) — THUDM · living doc · Bitwise reruns from SGLang deterministic inference plus Megatron deterministic mode. Requires removing FA3 and setting `NCCL_ALGO=Ring`.
- [Miles v0.1: Production-level post-training](https://www.lmsys.org/blog/2026-08-18-miles-v0-1) — RadixArk and partners · 2026-08-18 · Zero-KL alignment for the Qwen3 family, R3 routing replay, and bit-exact MXFP8/NVFP4 quantizer contracts.
- [Defending Against the Training–Inference Numeric Mismatch in RL (Especially Linear Attention) — and Whether It Helps Async RL](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) — Yichuan Wang, with the TorchTitan team · 2026-08 · Batch-invariant kernels plus a shared model definition give a 0 logprob gap for Gated DeltaNet (linear attention) on Qwen3.5-9B and 35B-A3B. Chunked (training) and recurrent (decode) GDN are different algorithms, not just different kernels. "BI costs 2–3× trainer throughput", about 5× on a terminal-agent task; "the unified model is free". Accuracy gains mostly fall within noise; recommends BI as a debugging tool.
- [IsoExec: Unified Execution to Eliminate Trainer-Inference Mismatch in SkyRL](https://vllm.ai/blog/2026-08-21-isoexec) — Alexander Jiang and SkyRL Team · 2026-08-21 · An execution contract (kernel, accumulation dtype, reduction order) with SHA-256 digests, parallelism-invariant kernels, and one batch-invariant model shared by vLLM and Megatron. On Qwen3.5-35B-A3B (8×H100) the full step went from 1224.6 s to 1534.0 s: 25.3% overhead (generation +31.3%, training +18.6%). Code: [SkyRL-IsoExec](https://github.com/zanderjiang/SkyRL-IsoExec).
- [vime × RL-Kernel × AMD: Bitwise train–rollout consistency on ROCm](https://vllm.ai/blog/2026-09-14-rl-kernel-v0-1-0-train-rollout-bitwise-consistency) — RL-Kernel, vime and AMD teams · 2026-09-14 · Megatron ↔ vLLM on 8×MI300X (Qwen3-8B GRPO): zero mismatches over 200 steps. No overhead figure. Code: [RL-Align/RL-Kernel](https://github.com/RL-Align/RL-Kernel).
- [vime: Reproducibility](https://docs.vllm.ai/projects/vime/en/latest/advanced/reproducibility.html) — vLLM · living doc · GLM-5 (MLA + DSA) train/rollout logprob gate below 1e-6. Needs batch-invariant DeepGEMM/DeepEP and a frozen indexer.
- [Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven](https://vllm.ai/blog/2026-10-14-train-inference-consistency) — Yan Bai (NVIDIA), Ao Shen (Inferact) · 2026-10-14 · The training forward calls vLLM's own kernels. DeepSeek-V4-Flash has diff = 0 for 100 steps at 91.3% of baseline rollout throughput; Nemotron NVFP4 hybrid at 87.5%. About 10% end-to-end overhead in a verl preview. Packaged as an agent skill ([vLLM #61551](https://github.com/vllm-project/vllm/pull/61551)). Limits: TP = 1, no spec decode.
- [VeXact: verl zero-mismatch rollout](https://github.com/verl-project/vexact) — verl-project · 2026-03 · A bitwise-aligned HF-based rollout engine sharing model definitions with FSDP. On Qwen3-30B-A3B, vLLM's rollout-correction KL explodes after about 300 steps; VeXact stays at 0.
- [verl: Full determinism for reproducible RL training](https://verl.readthedocs.io/en/latest/advance/determinism.html) — verl · living doc · `fsdp_config.batch_invariant=true` exports `VLLM_BATCH_INVARIANT=1`. On SM80 all GEMMs run in Triton, so steps are slower.
- [Unlocking Stable On-Policy RL on TPU with Bitwise-Zero Train–Inference Mismatch](https://developers.googleblog.com/unlocking-stable-on-policy-rl-on-tpu-with-bitwise-zero-traininference-mismatch/) — Zhang, Chai, Wang, Bao, Yu (Google) · 2026-11-16 · "Zero-TIM" in JAX/Pallas/Tunix: batch-invariant Pallas kernels, fixed-order cross-shard sums, custom VJPs (Qwen3-8B, 64 v5p chips). No overhead figure; MoE and linear attention still in progress.
- [Detent: Batch-Invariant LLM Serving and RL Rollouts at Near-Stock Speed](https://arxiv.org/abs/2611.10611) — Xin et al. · 2026-11-07 · Pins the batch-dependent values that kernel selection reads. Latency 1.04–1.17× of stock vs 1.67–10.3× for deterministic modes; the trainer matches every rollout logprob at 1.03× stock rollout time per token.
- [Fireworks: Why alignment of numerics and MoE routing matter](https://fireworks.ai/blog/reinforcement-learning-why-alignment-of-numerics-and-MoE-routing-matter) — Fireworks AI · 2026-09-30 · GLM-5.2 countdown task: an unaligned KL of about 0.013 made safeguards discard about 45% of each batch and reward collapsed. Aligned KL = 0 stayed stable.
- [Fireworks: Training-inference parity in MoE models: where numerics drift](https://fireworks.ai/research/moe-training-inference-parity) — Fireworks AI · 2026-03-10 · All-reduce topology, fused norm and MoE-combine fusions each shift numerics. Uses a k3 < 0.001 pass gate.
- [NeMo RL: Adding new models (logprob error check)](https://docs.nvidia.com/nemo/rl/nightly/adding-new-models.html) — NVIDIA · living doc · Gate: mean exp(|Δlogprob|) ≤ 1.05. Lists mismatch causes: kernels, prefill vs decode, TP, cumsum, refit, batch size.

**Statistical corrections (IS / TIS / MIS / masking)**

- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl) — Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, Jianfeng Gao · 2025-08-05 · Shows the vLLM vs FSDP gap makes RL off-policy. Introduces **Truncated Importance Sampling (TIS)**. Workshop version: [On the Rollout-Training Mismatch in Modern RL Systems](https://neurips.cc/virtual/2025/124259) (NeurIPS 2025 OPT). verl: [#2953](https://github.com/verl-project/verl/pull/2953).
- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda) — Jiacai Liu, Yingru Li et al. · 2025-09-17 · Sequence-level and geometric masked IS (**MIS**, Geo-RS), integrated in verl 0.6.0. Follow-up theory: [Part 1: Why Mismatch Breaks LLM-RL](https://yingru.notion.site/Part-1-Why-Mismatch-Breaks-LLM-RL-295211a558b780b8bb4cc8e367d906d0) (2025-10-30) bounds bias by TV distance and variance by χ².
- [Small Leak Can Sink a Great Ship — IcePop](https://ringtech.notion.site/icepop) — Ant Group Ling Team · 2025-09-19 · Double-sided token masking of the train/infer ratio for MoE. Paper: [Every Step Evolves (Ring-1T)](https://arxiv.org/abs/2510.18855) (2025-10-21). Successor: [KPop](https://ringtech.notion.site/kpop) (2026-05-26), which uses a binary-KL band.
- [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) — Zheng et al. (Qwen) · 2025-07-24 · The sequence-level ratio tolerates precision gaps between engines and removes the need for routing replay in MoE.
- [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers (R3)](https://arxiv.org/abs/2510.11370) — Ma et al. · 2025-10-13 · Rollout Routing Replay records the inference engine's expert routing and replays it in training.
- [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374) — Zheng et al. (Qwen) · 2025-12-01 · The token-level surrogate is valid only when **both** train–inference discrepancy and policy staleness are small. Justifies IS, clipping and routing replay.
- [Soft Adaptive Policy Optimization (SAPO)](https://arxiv.org/abs/2511.20347) — Gao et al. (Qwen) · 2025-11-25 · A soft gate replaces hard clipping for high-variance MoE token ratios.
- [Trust Region Masking for Long-Horizon LLM Reinforcement Learning](https://arxiv.org/abs/2512.23075) — Yingru Li et al. · 2025-12-28 · Classical trust-region bounds scale as O(T²) under rollout/trainer divergence; proposes masking-based bounds.
- [Dynamic Vocabulary Pruning: Stable LLM-RL by Taming the Tail](https://arxiv.org/abs/2512.23087) — Yingru Li et al. · 2025-12-28 · The logprob mismatch bound scales with (1 − p); prunes the vocabulary tail.
- [Adaptive Layerwise Perturbation (ALP)](https://arxiv.org/abs/2603.19470) — Ye et al. · 2026-03-19 · Learnable layer perturbations correct both staleness and mismatch.
- [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807) — Marek, Ryabinin et al. · 2026-09-17 · An additive correction cancels persistent train/infer drift. Argues that fully removing mismatch "would come at a major cost to rollout efficiency". Follow-up: [Regularized Score Centering](https://arxiv.org/abs/2611.21791) (2026-11-16).
- [Rethinking Training-Inference Mismatch in LLM RL: Where It Arises and How to Correct It (CIS)](https://arxiv.org/abs/2609.32444) — Yu et al. · 2026-09-26 · Models mismatch as a logit displacement; calibrated IS with a confidence-aware cap.
- [Router Drift Control (RDC)](https://arxiv.org/abs/2610.21682) — Jin et al. · 2026-10-15 · Argues that R3 biases gradients and rejects high-routing-drift responses instead. +5.5 over TIS (AIME avg, async).
- [verl: Rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) — Yingru Li (verl) · living doc · Token/seq TIS, seq-MIS, Geo-RS, K3-RS, bypass mode and score centering in one framework. Lists "policy mismatch" and "temporal lag" as separate scenarios.
- [NeMo RL: ProRLv2 guide](https://docs.nvidia.com/nemo/rl/latest/guides/prorlv2.html) — NVIDIA · living doc · `tis`, `icepop` and `seq-mask-tis` corrections with example bounds.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — OpenRLHF · living doc · `--algo.advantage.is_correction_*` flags for ICEPOP, seq-mask-TIS, TIS, binary-KL and TV gating.
- [Tinker: Importance-sampling loss](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/) — Thinking Machines · living doc · Records sampler logprobs as the correction term.
- [prime-rl: IPO loss](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/ipo.md) — Prime Intellect · living doc · Masks tokens with |π_train − π_infer| > ε. In a GLM-4.5-Air SWE ablation, IcePop's KL "explodes around step 1.2k" after masking only 0.003% of tokens.

**Precision and numerics fixes**

- [MiniMax-M1](https://arxiv.org/abs/2506.13585) — MiniMax · 2025-06-16 · §3.2: the train/infer probability mismatch "prevented reward growth". An **FP32 LM head** raised the train/infer probability correlation to about 0.99. verl port: [#8004](https://github.com/verl-project/verl/pull/8004).
- [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788) — Penghui Qi, Zichen Liu et al. (Sea AI Lab, NUS) · 2025-10-30 · BF16 rounding is the root cause; switching to FP16 largely removes the mismatch. Discussion: [OpenRLHF #1155](https://github.com/OpenRLHF/OpenRLHF/issues/1155).
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) — DeepSeek-AI · 2025-12-02 · §3.1: unbiased KL estimate, off-policy sequence masking, Keep Routing and Keep Sampling Mask.
- [GLM-5](https://arxiv.org/abs/2602.15763) — Zhipu · 2026-02-17 · IcePop, plus a deterministic top-k for the DSA indexer, because indexer replay at k = 2048 is impractical.
- [miles #975: Preserve fp32 `A_log` for Qwen3.5](https://github.com/radixark/miles/pull/975) — 2026-04-14 · Rounding GDN's `A_log` to BF16 broke Megatron ↔ SGLang parity.
- [Run High-Throughput RL Training with End-to-End FP8 Precision](https://developer.nvidia.com/blog/run-high-throughput-reinforcement-learning-training-with-end-to-end-fp8-precision/) — Huang et al. (NVIDIA) · 2026-04-20 · End-to-end FP8 has less mismatch than FP8 rollout alone; token TIS closes the gap to BF16. About 48% faster than BF16 on Qwen3-8B.
- [FP8 RL in SkyRL](https://www.anyscale.com/blog/fp8-reinfinforcement-learning-in-skyrl) — Anyscale · 2026-08-25 · Syncs the trainer's FP8 payloads directly to avoid dequantize/requantize drift.
- [FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning](https://arxiv.org/abs/2601.18150) — Qiu et al. · 2026-01-26 · Handles mismatch from FP8 rollouts and weight sync.
- [QaRL: Rollout-Aligned Quantization-Aware RL](https://arxiv.org/abs/2604.07853) — Gu et al. · 2026-04-09 · The trainer forward mimics the quantized rollout. +5.5 over quantized-rollout training on Qwen3-30B-A3B.
- [Towards Full Pipeline FP8 RL for LLMs](https://arxiv.org/abs/2609.22870) — Chen et al. · 2026-09-19 · Compounded FP8 noise distorts the IS ratio even with TIS.
- [FourSight: Stable FP4 RL Post-Training](https://arxiv.org/abs/2611.04887) — Wei et al. · 2026-11-03 · NVFP4 mismatch-aware sequence filtering, within 2.3% of FP4 step time.

**Async RL and staleness**

- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2505.24298) — Fu et al. (Ant, Tsinghua) · 2025-05-30 · Fully asynchronous RL with decoupled PPO; up to 2.77× speedup.
- [PipelineRL](https://arxiv.org/abs/2509.19128) — Piché et al. · 2025-09-23 · In-flight weight updates; about 2× faster learning.
- [prime-rl: Async training and staleness](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/algorithms.md) — Prime Intellect · living doc · Logs mismatch KL per log2 staleness bin. Bin 0 "holds only the numerical mismatch".
- [Prime Intellect RL glossary: Trainer–inference mismatch](https://www.primeintellect.ai/rl-glossary/trainer-inference-mismatch) — Prime Intellect · living doc · Mismatch "persists even with zero policy lag" and is separate from staleness.
- [Stable Asynchrony: Variance-Controlled Off-Policy RL (VCPO)](https://arxiv.org/abs/2602.17616) — Huang, Han et al. · 2026-02-19 · Stable at 128 steps off-policy; 2.5× speedup.
- [Stale but Stable: Staleness-Adaptive Trust Regions](https://arxiv.org/abs/2607.18722) — Yang et al. · 2026-07-21 · Attributes staleness to policy lag, engine delays and MoE routing; adapts the trust region to it.
- [Deconstructing Off-Policy Ratios: Entropy-Normalized Trust Regions for Async RL](https://arxiv.org/abs/2607.22186) — Zhao et al. · 2026-07-24 · Low-entropy tokens amplify mismatch noise.
- [How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?](https://arxiv.org/abs/2611.11450) — Yu et al. · 2026-11-08 · Puts quantization and async mismatch on one scale for on-policy distillation.

**Analyses: separating numeric mismatch from staleness**

- [On the Origin and Evolution of Training-Inference Mismatch in LLM RL](https://arxiv.org/abs/2610.40945) — Li, Zhou, Lan et al. · 2026-10-29 · Splits the error exactly into numerical discrepancy (roughly constant) and policy lag (a feedback loop). IS fails once ESS falls below one half.
- [Diagnosing Training Inference Mismatch in LLM RL via a Zero-Mismatch Reference](https://arxiv.org/abs/2605.14220) — Zhong et al. (VeXact) · 2026-05-14 · Uses a zero-mismatch reference to show that tiny token-level differences alone can collapse RL.
- [Beyond Precision: Training-Inference Mismatch is an Optimization Problem](https://arxiv.org/abs/2602.01826) — Zhang, Yingru Li et al. · 2026-02-02 · Mismatch and gradient noise escalate together; LR decay helps.
- [slime: From policy gradients to train–rollout consistency](https://github.com/THUDM/slime/blob/main/docs/en/advanced/policy-mismatch.md) — THUDM · living doc · Tutorial with three policies (behavior q, p₀, p_θ) covering TIS, ICE-POP, R3 and score centering.

**Issues and PRs worth reading**

- [verl-omni #637: 0-diff train-rollout log-probs for the Qwen3-Omni thinker](https://github.com/verl-project/verl-omni/issues/637) — cr-gao · 2026-09-20 · An opt-in "HF numerics" mode gives 0/8192 mismatches. Costs "−37% against the CUDA-graph baseline on this 4-layer prototype" (10,250 → 6,467 tok/s, 1×H20); not profiled. Rollout half: [vllm-omni #8186](https://github.com/vllm-project/vllm-omni/pull/8186).
- [verl #8025: Raw log-prob mismatch metrics](https://github.com/verl-project/verl/pull/8025) — 2026-10-22 · The old `exp()`-based metric hid 1-ULP differences.
- [verl #8026: Batch-invariant actor forward](https://github.com/verl-project/verl/pull/8026) — 2026-11-11 (merged).
- [verl #7688: DeepSeek-V4 exact 0-diff true-on-policy RL on Megatron + vLLM](https://github.com/verl-project/verl/pull/7688) — 2026-09-02 (open).
- [vLLM #48305: RFC: Training–inference consistency for RL](https://github.com/vllm-project/vllm/issues/48305) — 2026-07-11 · Eight consistency dimensions. See also [#42259](https://github.com/vllm-project/vllm/issues/42259) (logprob semantics) and [#62365](https://github.com/vllm-project/vllm/pull/62365) (bitwise DeepSeek-V4.1-Flash with CUDA graphs).
- [slime #566](https://github.com/THUDM/slime/pull/566) / [SGLang #12058](https://github.com/sgl-project/sglang/pull/12058) — 2025-10-24 · "True on-policy" support.
- [SGLang #26359: True-on-policy contract definitions](https://github.com/sgl-project/sglang/pull/26359) — 2026-05-26 (open).
- [SkyRL #2201: Full-distribution comparison for IsoExec validation](https://github.com/NovaSky-AI/SkyRL/issues/2201) — 2026-09-12.
- [SkyRL #699: Track inference vs trainer logprob mismatch](https://github.com/NovaSky-AI/SkyRL/issues/699) — 2025-11-24.
- [AReaL #1405: IcePop and KPop masking](https://github.com/areal-project/AReaL/pull/1405) — 2026-06-15 (merged).

### Inference determinism

- [Non-determinism in GPT-4 is caused by Sparse MoE](https://152334h.github.io/blog/non-determinism-in-gpt-4/) — 152334H · 2023-08-05 · Hypothesis: requests in one batch compete for expert capacity, so one output depends on the others. The author says it may be "completely wrong".
- [Non-Determinism of "Deterministic" LLM Settings](https://arxiv.org/abs/2408.04667) — Atil, Aykent, Chittams et al. · 2024-08-06 · 5 LLMs × 8 tasks × 10 runs at T = 0: accuracy varies by up to 15%, and the best-to-worst gap reaches 70%.
- [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501) (v1: "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning") — Yuan, Li, …, Zirui Liu · 2025-06-11 · Batch size, GPU count and GPU type cause up to 9% accuracy swings in BF16 greedy decoding. **LayerCast**: 16-bit weights with FP32 compute.
- [Beyond Reproducibility: Token Probabilities Expose LLM Nondeterminism](https://arxiv.org/abs/2601.06118) — Fu, …, Reviriego, Liu · 2026-01-03 · Nondeterminism matters most for tokens with probability between 0.1 and 0.9.
- [LLM-42: Enabling Determinism in LLM Inference with Verified Speculation](https://arxiv.org/abs/2601.17768) — Gond, Kamath, Ramjee, Panwar (Microsoft) · 2026-01-25 · Fast nondeterministic decode followed by fixed-shape verification and rollback. Overhead is paid only for traffic that needs determinism.
- [MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference](https://arxiv.org/abs/2605.30218) — Chu, Zhou, Zhang · 2026-05-28 · Batch-induced token flips occur on 0.3–1.3% of decode steps. Verification triggered at 15–19% of steps halves LLM-42's latency cost.
- [CoRun: Padding is Simple and Efficient for Deterministic LLM Inference](https://arxiv.org/abs/2608.14376) — Zhao et al. · 2026-08-14 · Reports that batch-invariant kernels more than double latency and cut throughput by up to 74%. Padding gives 15–324% more throughput.
- [Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference](https://arxiv.org/abs/2609.38981) — Qin, Du, Zhang, Lentz, Zhuo · 2026-09-30 · Shows that vLLM's and SGLang's deterministic modes still differ under some execution variations. Proves determinism with Verus.
- [Correctness Forensics for Batch Speculative Decoding](https://arxiv.org/abs/2510.22876) — Zhang et al. (eBay) · 2025-10-26 · Ragged tensors corrupt outputs in some batched speculative decoders. EXSPEC reaches 95% exact match; the rest is attributed to floating-point nondeterminism.

**Verifiable inference: who needs exact bits**

- [TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference](https://arxiv.org/abs/2501.16007) — Ong et al. (Prime Intellect) · 2025-01-27 · Hashes top-k activations and tolerates nondeterminism instead of requiring exact bits. 258 bytes per 32 tokens.
- [Verde: Verification via Refereed Delegation for Machine Learning Programs](https://arxiv.org/abs/2502.19405) — Arun et al. (Gensyn) · 2025-02-26 · Bisects a dispute to the first diverging operator. **RepOps** fixes floating-point operation order for bitwise reproducibility across hardware. [Blog](https://www.gensyn.ai/research/verde-a-verification-system-for-machine-learning-over-untrusted-nodes).
- [EigenAI: Deterministic Inference, Verifiable Results](https://arxiv.org/abs/2602.00182) — Alves et al. · 2026-01-30 · Bit-exact inference on a fixed GPU architecture plus re-execution in a TEE: verification is a byte-equality check.
- [Bit-Exact AI Inference Verification Without Performance Tradeoffs](https://arxiv.org/abs/2606.00279) — Naci Cankaya · 2026-05-29 · vLLM and HF are "deterministic but not invariant". Bitwise recomputation works across GPU variants as long as atomics are avoided.
- [VeriLLM](https://arxiv.org/abs/2509.24257) (2025-09-29) · [TensorCommitments](https://arxiv.org/abs/2602.12630) (2026-02-13) · [Proof of Sampling](https://arxiv.org/abs/2405.00295) (2024-05-01) — Verification protocols built on recomputation or commitments; see each abstract for its overhead.

### Framework migration, upgrade equivalence, and CI

**Checkpoint conversion and port parity**

- [Megatron Core: Llama / Mistral support](https://docs.nvidia.com/megatron-core/developer-guide/latest/llama_mistral.html) — NVIDIA · living doc · Megatron and HF are not numerically identical (TE RMSNorm dtype, fused QKV, RoPE). Llama-2 benchmark drift: 0.15% mean, 0.79% max.
- [Megatron-Bridge: Parity-testing skill](https://www.skills.sh/nvidia-nemo/megatron-bridge/parity-testing) — NVIDIA NeMo · living doc (mirror) · Order of checks: weight round-trip, HF ↔ Megatron logit comparison, then generation.
- [MaxText: Model bring-up](https://maxtext.readthedocs.io/en/latest/guides/model_bringup.html) — Google · living doc · `forward_pass_logit_checker.py` against HF or golden logits. Target `--max_kl_div=1e-4`.
- [A Hugging Face Accelerate Story of Multiple Backends: FSDP and DeepSpeed](https://huggingface.co/blog/deepspeed-to-fsdp-and-back) — Hugging Face · 2024-06 · FSDP and DeepSpeed losses diverged because DeepSpeed upcasts master weights to FP32. Fixed in Accelerate 0.30.
- [TorchTitan](https://arxiv.org/abs/2410.06511) — Liang et al. (Meta) · 2024-10 · PyTorch-native reference stack used in the cross-framework reproductions above.

**CI and golden-value tests**

- [vLLM: Model correctness tests](https://docs.vllm.ai/en/latest/contributing/model/tests/) — vLLM · living doc · `check_outputs_equal` (exact text vs HF) and `check_logprobs_close` (mutual top-k).
- [verl #8025](https://github.com/verl-project/verl/pull/8025) — A reminder to measure raw logprob differences; `exp()`-based metrics hid 1-ULP differences.

**Compiler and upgrade numerics**

- [Major updates to the after-AOT accuracy minifier](https://dev-discuss.pytorch.org/t/major-updates-to-the-after-aot-accuracy-minifier/1241) — Edward Yang · 2023-05 · The torch.compile accuracy minifier now saves real inputs and adds an `analyze` mode that checks nondeterminism against eager and fp64.
- [torch.compile troubleshooting (accuracy minifier)](https://docs.pytorch.org/docs/main/torch.compiler_troubleshooting_old.html) — PyTorch · living doc · `TORCHDYNAMO_REPRO_AFTER="aot"` with `TORCHDYNAMO_REPRO_LEVEL=4` bisects to the smallest subgraph whose numerics diverge.

**Differential testing of DL frameworks and compilers**

- [CRADLE: Cross-Backend Validation to Detect and Localize Bugs in Deep Learning Libraries](https://hvpham.github.io/publication/2019-ICSE-CRADLE) — Pham, Lutellier, Qi, Tan · ICSE 2019 · 12 bugs and 104 inconsistencies across TF, CNTK and Theano.
- [Audee: Automated Testing for Deep Learning Frameworks](https://conf.researchr.org/details/ase-2020/ase-2020-papers/37/Audee-Automated-Testing-for-Deep-Learning-Frameworks) — Guo et al. · ASE 2020 · Cross-framework checks with causal localization; 26 new bugs.
- [LEMON: Deep Learning Library Testing via Effective Model Generation](https://github.com/Jacob-yen/LEMON) — Wang et al. · ESEC/FSE 2020 · Model mutation that amplifies inconsistencies; 24 new bugs.
- [NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers](https://arxiv.org/abs/2207.13066) — Liu et al. · ASPLOS 2023 · Differential testing of compiled vs eager models; 72 bugs (TVM, TensorRT, ORT, PyTorch).
- [Tzer](https://arxiv.org/abs/2202.09947) (2022-02) · [FreeFuzz](https://arxiv.org/abs/2201.06589) (2022-01) · [TitanFuzz](https://arxiv.org/abs/2212.14834) (ISSTA 2023) — Coverage-guided, API-level and LLM-generated fuzzing of DL stacks.
- [Deep Learning Framework Testing via Model Mutation: How Far Are We?](https://arxiv.org/abs/2506.17638) — Mu et al. · 2025-06 · Generic mutation operators produce many false positives.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al. · ICLR 2018 · The FP32-master-weights and loss-scaling baseline that most parity tests assume.

### Hardware faults and silent data corruption (SDC)

**Fleet studies**

- [Silent Data Corruptions at Scale](https://arxiv.org/abs/2102.11245) — Dixit et al. (Facebook) · 2021-02 · Defect types and a real datacenter case. [Blog](https://engineering.fb.com/2021/02/23/data-infrastructure/silent-data-corruption/): a core returned 0 from `math.pow`.
- [Cores that don't count](https://research.google/pubs/cores-that-dont-count/) — Hochschild et al. (Google) · HotOS 2021 · "Mercurial" cores; calls for detection, isolation and tolerance.
- [Detecting silent data corruptions in the wild](https://arxiv.org/abs/2203.08989) — Dixit et al. (Meta) · 2022-03 · Out-of-production Fleetscanner vs in-production Ripple, over 3+ years.
- [Understanding Silent Data Corruptions in a Large Production CPU Population](https://doi.org/10.1145/3600006.3613149) — Wang et al. (Alibaba Cloud et al.) · SOSP 2023 · Over 1M CPUs; floating-point and vector units are vulnerable.
- [How Meta keeps its AI hardware reliable](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/) — Dixit, Sankar (Meta) · 2025-07-22 · Fleetscanner, Ripple and Hardware Sentinel. SDC rate is now about 1 per 1,000 devices.
- [Revisiting Reliability in Large-Scale Machine Learning Research Clusters](https://arxiv.org/abs/2410.21680) — Kokolis et al. (Meta) · 2024-10 · 11 months, 4M jobs, failure taxonomy.

**SDC in LLM training**

- [Understanding and Mitigating Hardware Failures in Deep Learning Training Systems](https://doi.org/10.1145/3579371.3589105) — He et al. (UChicago, Google) · ISCA 2023 · How transient and permanent faults affect training; lightweight detection and recovery.
- [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) — Google · 2023-12 · Expects SDC "every week or two". Uses deterministic replay to isolate bad hardware.
- [Gemini 2.5](https://arxiv.org/abs/2507.06261) — Google · 2025-07 · Split-phase SDC detection: deterministic replay of suspicious steps plus per-device checksums. About 0.25% of steps are replayed; about 6% of those were real corruption.
- [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng) — Jiang et al. (ByteDance) · NSDI 2024 · Deep-stack diagnosis and fault tolerance at 12,288 GPUs.
- [Robust LLM Training Infrastructure at ByteDance (ByteRobust)](https://arxiv.org/abs/2509.16293) — Wan et al. · 2025-09 · Fast fault isolation; 97% ETTR over 3 months on 9,600 GPUs.
- [Understanding Silent Data Corruption in LLM Training](https://arxiv.org/abs/2502.12340) — Ma, Pei, Lausen, Karypis · ACL 2025 · Compares real unhealthy nodes with healthy ones under deterministic XLA. Mostly small perturbations that can still steer training to a different optimum.
- [Exploring Silent Data Corruption as a Reliability Challenge in LLM Training](https://arxiv.org/abs/2604.00726) — Altenbernd, Wiesner, Kao · CCGrid 2026 · Fault injection into MMA instructions; detection plus step recomputation.
- [LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training](https://arxiv.org/abs/2604.10390) — Tyagi et al. (NVIDIA) · SC 2026 · 7,664 RTL-fault runs in Megatron-LM.
- [The Anatomy of Silent Data Corruption](https://arxiv.org/abs/2605.04213) — Tung et al. (NVIDIA) · DSN 2026 · NaN/Inf is only about 1% of SDC outcomes, so NaN guards are not enough.
- [TrainSDC: Characterizing and Mitigating SDC in LLM Training](https://arxiv.org/abs/2608.30769) — Xia et al. · 2026-08 · Q/K-path faults persist. Mitigations cost 1.65–6.76% overhead.
- [SDCs in the Wild: Characterizing and Diagnosing SDC-Defective GPUs in Production LLM Training (SDCHunter)](https://www.usenix.org/conference/osdi26/presentation/zheng) — Zheng et al. (SJTU, ByteDance Seed) · OSDI 2026 · 23 defective GPUs. Synthetic benchmarks miss more than 60% of them, and ECC doesn't catch them. Replays the training workload on two DP replicas; mitigated 40 incidents.
- [Safeguarding LLM Training at Scale: Online SDC Detection and Insights from 35 Million GPU Hours (AEGIS)](https://www.usenix.org/conference/osdi26/presentation/lei) — Lei et al. (Tsinghua, ByteDance) · OSDI 2026 · Two-stage online detection: 18 SDC incidents, 13 faulty GPUs, 0.86% overhead.
- [Detecting and recovering from silent data corruption on Amazon SageMaker HyperPod](https://aws.amazon.com/blogs/machine-learning/detecting-and-recovering-from-silent-data-corruption-on-amazon-sagemaker-hyperpod/) — AWS · 2026-11 · Ring-replay audit with bitwise loss and gradient fingerprints, plus weight-fingerprint majority vote. Needs at least 3 nodes and deterministic kernels.
- [ATTNChecker: Highly-Optimized Fault Tolerant Attention for LLM Training](https://arxiv.org/abs/2410.11720) — Liang et al. · PPoPP 2025 · ABFT for attention; about 7% overhead.

**Framework tooling**

- [Megatron Core: Rerun state machine](https://docs.nvidia.com/megatron-core/developer-guide/0.20.0/apidocs/core/core.rerun_state_machine.html) — NVIDIA · living doc · Reruns a step on the same GPU, then on another, to classify a result as correct, transient or persistent. Needs deterministic execution. [NeMo resiliency](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html) marks it alpha.
- [DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/dcgm-diagnostics.html) — NVIDIA · living doc · Intrusive diagnostics that check for compute miscompares; drain the node first. SDCHunter reports that synthetic tests miss most SDC GPUs.

## Tools and debuggers

**Find where two runs first diverge**

- [Verde / RepOps](https://arxiv.org/abs/2502.19405) — Gensyn · 2025-02 · Bisects a dispute to the first diverging operator; see [Verifiable inference](#inference-determinism).
- [TTrace: Lightweight Error Checking and Diagnosis for Distributed Training](https://arxiv.org/abs/2506.09280) — Jiang et al. · 2025-06 · Differential testing of intermediate tensors against a single-device reference, with derived floating-point tolerances, since the authors consider bitwise matching impossible. Found 11 known and 3 new Megatron-LM bugs.
- [OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming) — Ziming Zhou, Yinjie Zhao (U. Michigan); Hang Zhu, Wenxiao Wang, Zhihao Bai, Yun Zhang, Shuguang Wang, Haibin Lin (ByteDance Seed); Peng Huang (U. Michigan) · deployed at ByteDance since 2025-10; OSDI 2026 · The debugging primitive: the first bitwise divergence across training stacks. Finds model-level operator boundaries that stay stable across training stacks and records lightweight per-operator fingerprints (XOR of the raw output bits). A schedule-tolerant mapper then finds the longest bitwise-matching prefix; the first mismatch is where diagnosis starts. Benign nondeterminism is controlled first, so the rest can be required to match exactly. Diagnosed 20+ production issues at ByteDance, including kernel races and SDCs, and cut debugging "from days to minutes". Code: [OrderLab/OpGuard](https://github.com/OrderLab/OpGuard) · [website](https://orderlab.io/OpGuard/).
- [PyTorch DebugMode](https://docs.pytorch.org/tutorials/recipes/debug_mode_tutorial.html) — Pawakapan, Yu (PyTorch) · PyTorch ≥ 2.10 (prototype) · Dispatch-level op log with `log_tensor_hashes`. Diff eager and compiled logs to find the first diverging op.
- [Ekka: Automated Diagnosis of Silent Errors in LLM Inference](https://arxiv.org/abs/2606.04594) — Gu, …, Kasikci · ICML 2026 · Aligns intermediate states between a serving framework and a reference; 80% pass@1 diagnosis.
- [Cross-Stack Validation of Language-Model Training: A Clinical Fine-Tuning Case Study](https://arxiv.org/abs/2608.24267) — Tran, Dang · 2026-08-25 · An independent Zig stack serves as a differential oracle for PyTorch; tolerance-based. Found 17 faults.
- [Megatron Core fingerprint tracing](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) — NVIDIA · 2026-10-06 · Per-rank `torch.hash_tensor` traces; see [Training frameworks](#training-frameworks).

**Invariants and checks**

- [TrainCheck: Training with Confidence](https://www.usenix.org/conference/osdi25/presentation/jiang) — Jiang, Zhou, Xu, Liu, Xu, Huang (U. Michigan) · OSDI 2025 · Infers training invariants automatically. Caught 18 of 20 real silent errors within one iteration and found 6 new library bugs.
- [Megatron-LM NaN / large-grad checks](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py) — NVIDIA · living code · NaN checks on loss and grad are on by default; also `--check-for-large-grads` and the rerun state machine.
- [tf.debugging.enable_check_numerics](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics) — Google · living doc · Errors on the first op that outputs Inf/NaN.
- [JAX checkify](https://docs.jax.dev/en/latest/debugging/checkify_guide.html) — Google · living doc · jit-compatible NaN and division checks; "can be expensive".

**Layer-wise comparison**

- [Hugging Face: `model_addition_debugger_context`](https://huggingface.co/docs/transformers/en/internal/model_debugging_utils) — Hugging Face · living doc · Logs every module's inputs and outputs for diffing two implementations. May miss external kernels.
- [Transformer Engine precision debug tools (nvdlfw_inspect)](https://docs.nvidia.com/deeplearning/transformer-engine/debug/1_getting_started.html) — NVIDIA · living doc · Per-GEMM tensor stats, and running selected GEMMs in high precision.
- [Polygraphy: Debugging accuracy](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/debug_accuracy.md) — NVIDIA · living doc · Layer-wise ONNX-RT vs TensorRT comparison, then `debug reduce`. Marking outputs can change optimizations and hide the bug.
- [PyTorch Numeric Suite](https://docs.pytorch.org/docs/2.8/torch.ao.ns._numeric_suite.html) — PyTorch · living doc (prototype) · Float vs quantized activation comparison.
- [NNsight and NDIF](https://arxiv.org/abs/2407.14561) — Fiotto-Kaufman et al. · 2024-07 · Read or intervene on any activation; usable as a layer-diff harness.
- [NVIDIA Compute Sanitizer (racecheck)](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) — NVIDIA · living doc · Detects shared-memory hazards. Races are one root cause of nondeterministic kernels.

## What exactness costs

Overheads as stated by each source. Setups differ, so compare with care.

<!-- costs:start -->
<!-- generated from data/timeline.json by scripts/build.py; edit the JSON, not this table -->

| Work | Setting | Stated cost |
|---|---|---|
| [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Batch-invariant vLLM, Qwen3-8B | 26 s → 42 s (55 s unoptimized); matmul about 20% slower than cuBLAS |
| [SGLang](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) | Deterministic inference | 34.35% average slowdown |
| [vLLM × TorchTitan](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) | Bitwise RL, Qwen3-1.7B | 2.4× slower |
| [Yichuan Wang](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) | Batch-invariant GDN trainer, Qwen3.5 | 2–3× trainer throughput (about 5× on a terminal-agent task) |
| [IsoExec / SkyRL](https://vllm.ai/blog/2026-08-21-isoexec) | Qwen3.5-35B-A3B, 8×H100 | 25.3% per RL step (generation +31.3%, training +18.6%) |
| [verl-omni #637](https://github.com/verl-project/verl-omni/issues/637) | HF-numerics rollout, 4-layer prototype | −37% rollout throughput |
| [vLLM 2026-10](https://vllm.ai/blog/2026-10-14-train-inference-consistency) | DeepSeek-V4-Flash, trainer calls vLLM kernels | 91.3% of baseline rollout throughput; about 10% end to end (verl preview) |
| [Detent](https://arxiv.org/abs/2611.10611) | Batch-invariant serving | 1.04–1.17× stock latency |
| [CoRun](https://arxiv.org/abs/2608.14376) | Survey of batch-invariant kernels | Up to 74% lower throughput, more than 2× latency |
| [DASH](https://arxiv.org/abs/2601.21824) | FA3 deterministic backward | Up to 37.9% throughput loss |
| [Megatron Core](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) | Bitwise pretraining, 3,072 GPUs | About 17% → 1.5% after optimization |
| [AEGIS](https://www.usenix.org/conference/osdi26/presentation/lei) | Online SDC detection | 0.86% |

<!-- costs:end -->

*Our reading:* the cost is falling fast, from 2× to under 10%, as exactness moves from wrapping kernels to designing them for it. Meanwhile, most RL stacks ship statistical corrections because they are free, and treat exact bits as a debugging mode.

## Timeline

<!-- timeline:start -->
<!-- generated from data/timeline.json by scripts/build.py; edit the JSON, not this table -->

★ Bold rows are the key milestones.

| Date | Milestone |
|---|---|
| [2017-10](https://training-alignment.github.io/awesome-bitwise-alignment/#mixed-precision) | [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al. (NVIDIA, Baidu): Sets the FP16 + FP32-master-weights baseline that most parity tests assume. |
| [2018-09](https://training-alignment.github.io/awesome-bitwise-alignment/#deterministic-drl) | [Deterministic deep RL, one source at a time](https://arxiv.org/abs/1809.05676) — Nagarajan, Warnell, Stone: A fully deterministic DQN; adding back any single nondeterminism source, GPU ops included, changes results. |
| [2019-05](https://training-alignment.github.io/awesome-bitwise-alignment/#cradle) | [CRADLE: cross-backend validation](https://hvpham.github.io/publication/2019-ICSE-CRADLE) — Pham et al. · ICSE ’19: Differential testing across TF, CNTK and Theano found 12 bugs and 104 inconsistencies. |
| [2020-01](https://training-alignment.github.io/awesome-bitwise-alignment/#tf-deterministic-ops) | [TF_DETERMINISTIC_OPS](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0) — TensorFlow 2.1: Deterministic cuDNN convolutions and bias_add behind one environment variable. |
| [2020-09](https://training-alignment.github.io/awesome-bitwise-alignment/#variance-study) | [Variance in DL training](https://hvpham.github.io/publication/2020-ASE-Variance) — Pham et al. · ASE ’20: Implementation-level nondeterminism alone moves accuracy by up to 2.9%. |
| [2020-10](https://training-alignment.github.io/awesome-bitwise-alignment/#pytorch-set-deterministic) | [torch.set_deterministic (beta)](https://github.com/pytorch/pytorch/releases/tag/v1.7.0) — PyTorch 1.7: The first PyTorch switch that errors on known nondeterministic ops. |
| [2021-02](https://training-alignment.github.io/awesome-bitwise-alignment/#meta-sdc) | [Silent Data Corruptions at Scale](https://arxiv.org/abs/2102.11245) — Meta: Silicon defects silently corrupt computation across a production fleet; the same code gives different bits. |
| [2021-03](https://training-alignment.github.io/awesome-bitwise-alignment/#pytorch-deterministic-algorithms) | [torch.use_deterministic_algorithms](https://github.com/pytorch/pytorch/releases/tag/v1.8.0) — PyTorch 1.8: Errors on known nondeterministic ops. Same hardware and version only; no guarantee across releases. |
| [2021-06](https://training-alignment.github.io/awesome-bitwise-alignment/#cores-that-dont-count) | [Cores that don’t count](https://research.google/pubs/cores-that-dont-count/) — Google · HotOS ’21: “Mercurial” cores silently compute wrong results. Bits can differ even when the code is identical. |
| [2022-02](https://training-alignment.github.io/awesome-bitwise-alignment/#tf-op-determinism) | [tf.config.experimental.enable_op_determinism](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0) — TensorFlow 2.8: An API replaces the TF_DETERMINISTIC_OPS environment variable. |
| [2022-02](https://training-alignment.github.io/awesome-bitwise-alignment/#reproducible-training-icse22) | [Towards training reproducible DL models](https://arxiv.org/abs/2202.02326) — Chen et al. · ICSE ’22: Record-and-replay plus profile-and-patch reproduced 6 open-source models and 1 commercial model. |
| **[2022-04](https://training-alignment.github.io/awesome-bitwise-alignment/#palm-bitwise)** | **[PaLM: fully bitwise-reproducible training](https://arxiv.org/abs/2204.02311)** — Google · PaLM report: “The model is fully bitwise reproducible from any checkpoint,” via a bitwise-deterministic JAX + XLA + T5X stack and a data pipeline where each batch is a function of the step number only. ★ *Opened bitwise-reproducible LLM training: rerun any step from any checkpoint and get the same bits.* |
| [2022](https://training-alignment.github.io/awesome-bitwise-alignment/#opt-logbook) | [OPT-175B training logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) — Meta AI: A full public logbook of a 175B training run. |
| [2022](https://training-alignment.github.io/awesome-bitwise-alignment/#bloom-chronicles) | [BLOOM-176B chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) — BigScience: Records a BF16 optimizer bug that skipped grad clipping for TP > 1 and desynced layer norms. |
| [2023-08](https://training-alignment.github.io/awesome-bitwise-alignment/#gpt4-moe-nondeterminism) | [GPT-4 nondeterminism and sparse MoE](https://152334h.github.io/blog/non-determinism-in-gpt-4/) — 152334H (blog): Hypothesis: requests batched together compete for expert capacity. The author says it may be wrong. |
| [2023-11](https://training-alignment.github.io/awesome-bitwise-alignment/#openai-seed) | [seed + system_fingerprint](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter) — OpenAI: Determinism in a hosted API, but only “best effort”: outputs are mostly identical, not guaranteed. |
| [2023-12](https://training-alignment.github.io/awesome-bitwise-alignment/#gemini-replay) | [Deterministic replay to find bad chips](https://arxiv.org/abs/2312.11805) — Gemini report: SDC expected “every week or two”; replaying a step deterministically isolates the faulty hardware. |
| [2024-05](https://training-alignment.github.io/awesome-bitwise-alignment/#xla-deterministic-ops) | [xla_gpu_exclude_nondeterministic_ops](https://github.com/openxla/xla/pull/12520) — OpenXLA: Deterministic scatter, GEMM and convolution in XLA without turning off autotuning. |
| [2024-07](https://training-alignment.github.io/awesome-bitwise-alignment/#llama3) | [Loss parity across parallelism](https://arxiv.org/abs/2407.21783) — Llama 3 report: Comparing loss across parallel layouts found numerical bugs; 6 of 419 interruptions were SDC. |
| [2024-08](https://training-alignment.github.io/awesome-bitwise-alignment/#temp0-nondeterminism) | [Temperature 0 is not deterministic](https://arxiv.org/abs/2408.04667) — Atil et al.: Accuracy varies by up to 15% across runs at T = 0. |
| [2025-06-11](https://training-alignment.github.io/awesome-bitwise-alignment/#layercast) | [“Give Me FP32 or Give Me Death?” (LayerCast)](https://arxiv.org/abs/2506.09501) — Yuan et al.: Batch size and GPU count cause up to 9% accuracy swings in BF16 greedy decoding; FP32 compute fixes it. |
| [2025-06-16](https://training-alignment.github.io/awesome-bitwise-alignment/#minimax-fp32-head) | [FP32 LM head for RL](https://arxiv.org/abs/2506.13585) — MiniMax-M1 report: The train/infer probability mismatch “prevented reward growth” until the LM head ran in FP32. |
| [2025-07](https://training-alignment.github.io/awesome-bitwise-alignment/#traincheck) | [TrainCheck](https://www.usenix.org/conference/osdi25/presentation/jiang) — U. Michigan · OSDI ’25: Inferred training invariants caught 18 of 20 real silent errors within one iteration. |
| **[2025-08-05](https://training-alignment.github.io/awesome-bitwise-alignment/#tis)** | **[Your efficient RL is secretly off-policy](https://fengyao.notion.site/off-policy-rl)** — Feng Yao et al.: vLLM and FSDP disagree on token probabilities with the same weights. Correct it with truncated importance sampling. ★ *Opened train–inference mismatch in RL: numeric differences quietly make on-policy RL off-policy.* |
| **[2025-09-10](https://training-alignment.github.io/awesome-bitwise-alignment/#defeating-nondeterminism)** | **[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)** — Thinking Machines: The culprit is batch invariance. 80 unique temp-0 outputs become 1, and a matching trainer gives KL = 0. ★ *Opened batch invariance: the root cause of inference nondeterminism, and the path to exact RL.* |
| [2025-09-17](https://training-alignment.github.io/awesome-bitwise-alignment/#when-speed-kills-stability) | [When Speed Kills Stability](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda) — Jiacai Liu, Yingru Li et al.: Sequence-level and geometric masked importance sampling (MIS) against mismatch-driven collapse. |
| [2025-09-19](https://training-alignment.github.io/awesome-bitwise-alignment/#icepop) | [IcePop](https://ringtech.notion.site/icepop) — Ant Group Ling Team: Double-sided token masking of the train/infer ratio for MoE RL. |
| [2025-09-22](https://training-alignment.github.io/awesome-bitwise-alignment/#sglang-deterministic) | [Deterministic inference in SGLang](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) — LMSYS: Works with chunked prefill, CUDA graphs and radix cache, at a 34.35% average slowdown. |
| [2025-09-26](https://training-alignment.github.io/awesome-bitwise-alignment/#vllm-batch-invariance-pr) | [First batch-invariance PR in vLLM](https://github.com/vllm-project/vllm/pull/25603) — vLLM: Kernel-override plumbing that later became VLLM_BATCH_INVARIANT. |
| [2025-10-13](https://training-alignment.github.io/awesome-bitwise-alignment/#r3) | [Rollout Routing Replay (R3)](https://arxiv.org/abs/2510.11370) — Ma et al.: Records the inference engine’s MoE routing and replays it in training. |
| **[2025-10](https://training-alignment.github.io/awesome-bitwise-alignment/#opguard)** | **[OpGuard](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)** — ByteDance production · OSDI ’26: Operator-level fingerprints across training stacks report the first bitwise divergence. In production at ByteDance from October 2025; debugging went from days to minutes. ★ *Made the first bitwise divergence a debugging primitive: across training stacks, the first mismatch is where diagnosis starts.* |
| [2025-10-24](https://training-alignment.github.io/awesome-bitwise-alignment/#true-on-policy-slime) | [“True on-policy” in slime and SGLang](https://github.com/THUDM/slime/pull/566) — slime / SGLang: Rollout and training log-probs match exactly. |
| [2025-10-30](https://training-alignment.github.io/awesome-bitwise-alignment/#fp16-rl) | [Defeating the mismatch via FP16](https://arxiv.org/abs/2510.26788) — Qi, Liu et al. (Sea AI Lab): BF16 rounding is the root cause; switching to FP16 largely removes the mismatch. |
| [2025-11-10](https://training-alignment.github.io/awesome-bitwise-alignment/#vllm-torchtitan-bitwise) | [Bitwise on-policy RL](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) — vLLM × TorchTitan: Trainer and sampler agree bit for bit on Qwen3-1.7B, but “the bitwise RL run is 2.4x slower”. |
| [2025-11-19](https://training-alignment.github.io/awesome-bitwise-alignment/#miles) | [Miles](https://www.lmsys.org/blog/2025-11-19-miles) — RadixArk: “True on-policy” RL with zero KL for large MoE training. |
| [2025-11-21](https://training-alignment.github.io/awesome-bitwise-alignment/#tbik) | [Deterministic across TP sizes (TBIK)](https://arxiv.org/abs/2511.17826) — Zhang et al. · ICML ’26: Tree-based invariant kernels give identical bits across tensor-parallel sizes. |
| [2025-12-01](https://training-alignment.github.io/awesome-bitwise-alignment/#qwen-stabilizing-rl) | [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374) — Qwen: The token-level surrogate holds only when both train–inference discrepancy and staleness are small. |
| [2026-01-25](https://training-alignment.github.io/awesome-bitwise-alignment/#llm-42) | [LLM-42: verified speculation](https://arxiv.org/abs/2601.17768) — Microsoft Research: A fast nondeterministic decode, then verify and roll back; you pay only for traffic that needs determinism. |
| [2026-01-29](https://training-alignment.github.io/awesome-bitwise-alignment/#dash) | [DASH: deterministic attention scheduling](https://arxiv.org/abs/2601.21824) — SJTU · ICLR ’26: Recovers up to 1.28× of the throughput FA3 loses in deterministic backward. |
| [2026-05-14](https://training-alignment.github.io/awesome-bitwise-alignment/#vexact) | [A zero-mismatch reference (VeXact)](https://arxiv.org/abs/2605.14220) — verl: Shows that tiny token-level mismatch alone can collapse RL. |
| [2026-07](https://training-alignment.github.io/awesome-bitwise-alignment/#aegis) | [AEGIS: online SDC detection](https://www.usenix.org/conference/osdi26/presentation/lei) — Tsinghua · ByteDance · OSDI ’26: 18 SDC incidents and 13 faulty GPUs found at 0.86% overhead, over 35 million GPU hours. |
| [2026-07](https://training-alignment.github.io/awesome-bitwise-alignment/#sdchunter) | [SDCs in the Wild (SDCHunter)](https://www.usenix.org/conference/osdi26/presentation/zheng) — SJTU · ByteDance Seed · OSDI ’26: Synthetic benchmarks miss over 60% of SDC-defective GPUs; replaying the real workload catches them. |
| [2026-08](https://training-alignment.github.io/awesome-bitwise-alignment/#gdn-mismatch) | [Linear attention: an algorithmic mismatch](https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/) — Yichuan Wang: Chunked vs recurrent GDN are different algorithms. Bitwise parity costs 2–3× trainer throughput. |
| [2026-08-21](https://training-alignment.github.io/awesome-bitwise-alignment/#isoexec) | [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec) — SkyRL · vLLM blog: One execution contract for vLLM and Megatron, at 25.3% per-step overhead on Qwen3.5-35B-A3B. |
| [2026-09-02](https://training-alignment.github.io/awesome-bitwise-alignment/#olmo3-torchtitan) | [Olmo3 reproduced in TorchTitan](https://docs.pytorch.org/devlogs/distributed/2026-09-02-olmo3-reproduction/) — PyTorch: Matching RoPE, RMSNorm, fused QKV and grad-norm reduction order moved the port toward bitwise. |
| [2026-09-14](https://training-alignment.github.io/awesome-bitwise-alignment/#amd-bitwise-rl) | [Bitwise RL on AMD MI300X](https://vllm.ai/blog/2026-09-14-rl-kernel-v0-1-0-train-rollout-bitwise-consistency) — RL-Kernel · vime · AMD: Megatron and vLLM match for 200 steps on 8×MI300X. |
| [2026-09-20](https://training-alignment.github.io/awesome-bitwise-alignment/#verl-omni-637) | [HF-numerics rollout (verl-omni #637)](https://github.com/verl-project/verl-omni/issues/637) — verl-omni: 0-diff log-probs for the Qwen3-Omni thinker, at −37% rollout throughput on a 4-layer prototype. |
| [2026-10-06](https://training-alignment.github.io/awesome-bitwise-alignment/#megatron-bitwise-pretraining) | [Bitwise pretraining at 3,072 GPUs](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) — NVIDIA Megatron Core: Determinism tax down from ~17% to ~1.5%, with per-rank hash traces to localize divergence. |
| [2026-10-14](https://training-alignment.github.io/awesome-bitwise-alignment/#vllm-fast-bitwise-rl) | [Bitwise-consistent, high-performance RL](https://vllm.ai/blog/2026-10-14-train-inference-consistency) — vLLM blog: DeepSeek-V4-Flash: diff = 0 at 91.3% of baseline rollout throughput. Packaged as an agent skill. |
| [2026-10-29](https://training-alignment.github.io/awesome-bitwise-alignment/#dlc) | [Origin and evolution of the mismatch (DLC)](https://arxiv.org/abs/2610.40945) — Li, Zhou, Lan et al.: Splits the error into numerical discrepancy, roughly constant, and policy lag, a feedback loop. |
| [2026-11-07](https://training-alignment.github.io/awesome-bitwise-alignment/#detent) | [Detent: near-stock batch invariance](https://arxiv.org/abs/2611.10611) — Xin et al.: Batch-invariant serving at 1.04–1.17× stock latency. |
| [2026-11-16](https://training-alignment.github.io/awesome-bitwise-alignment/#tpu-zero-tim) | [Zero train–inference mismatch on TPU](https://developers.googleblog.com/unlocking-stable-on-policy-rl-on-tpu-with-bitwise-zero-traininference-mismatch/) — Google: Batch-invariant Pallas kernels and fixed-order cross-shard sums in JAX. |

<!-- timeline:end -->

## Not yet verified

These were found but could not be confirmed from a primary source. They are left out of the list above until someone confirms them. PRs welcome.

- **DeepSeek-V4 technical report** ([arXiv 2606.19348](https://arxiv.org/abs/2606.19348)). Secondary sources say it uses end-to-end batch-invariant, deterministic kernels (no split-K/split-KV, DeepGEMM instead of cuBLAS). The abstract does not say so, and the PDF was not read.
- **OCP "Silent Data Corruption in AI" whitepaper** ([v1.1](https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-ver-1-1-final-pdf/)). The PDF text could not be extracted, so its authors and dates are unconfirmed.
- **Anthropic's** wording that temperature 0 is not fully deterministic. It is widely quoted but not found on the current docs page.
- **Megatron-LM golden-value functional tests** and the nightly **determinism status / op catalog** pages. Seen only in mirrors and search snippets.
- **Power Up FSDP2 as a Flexible Training Backend for Miles** (LMSYS, 2025-12-03). The page returned 404.
- First-party determinism docs for **TensorRT-LLM, LMDeploy, llama.cpp, FlashInfer, DeepSpeed, Together, Mistral**. None found.
- Duncan Riach, "Determinism in Deep Learning" (GTC 2019, S9911). The slides could not be opened.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md). In short: link the primary source, give authors and date, write one line on what it does, and quote a cost only if the source states it. Run `python3 scripts/check_links.py README.md` before opening a PR.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related rights to this work. See [LICENSE](LICENSE).

## 中文简介

[English](#readme) · **简体中文** · [双语时间线动画](site/index.html)

本列表收集 **比特级对齐 (bitwise alignment)** 相关工作：确定性、batch invariance、训练与推理 (rollout/trainer) 对齐。按两种方式组织：一是各大实验室和框架（Thinking Machines、vLLM、SGLang、PyTorch、Megatron、TorchTitan、XLA 等）如何实现比特级可复现；二是按使用场景，包括预训练复现与调试、RL 训推一致（同步/异步）、推理确定性、框架迁移与升级等价性、硬件 SDC 检测。可交互的双语时间线动画见 [网站](site/index.html)（英文 / 简体中文 / 双语对照）。即将推出《训练对齐手册》（Training Alignment Handbook）：让训练栈逐比特一致的方案、检查清单与案例。另附调试工具（如用算子级指纹定位跨栈的第一个比特分歧）、代价对比表，以及标出各个“首创”工作的时间线。每条都附原始链接、作者、日期，以及来源中写明的代价与限制。
