# Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation

This repository provides code for the paper **“Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation”** (preliminary work; under review for the **ICML 2026 Workshop on Weight-Space Symmetries**).

The implementation builds on [nanoGPT](https://github.com/karpathy/nanoGPT) and follows the mHC / mHC-lite line of Hyper-Connection language models as in [**mHC-lite: You Don’t Need 20 Sinkhorn-Knopp Iterations**](https://arxiv.org/abs/2601.05732).

## Abstract

Hyper-Connections (HC) replace the single Transformer residual stream with multiple streams, introducing a permutation symmetry over stream indices. We study how this symmetry is resolved in practice: do streams specialize, or does the model collapse toward an effectively single-stream pathway? Using fine-grained diagnostics for HC-based language models, we trace how multi-stream representations are actually used. We find that after an early seeding stage residual mixing remains close to identity, so the model fails to use the main HC idea of exchanging information between streams. Moreover, both signal and interpretable features concentrate in a single pathway. Thus, the nominally multi-stream residual connection can underutilize its capacity, behaving much closer to a single-stream architecture. Finally, we show that breaking symmetry at stream initialization reduces dominant behavior and improves performance across mHC variants.

## Contributions

- We identify a stream-level failure mode in HC-style residuals: models with multiple symmetric streams can rely on one dominant stream.
- We show that collapse arises in mechanics and semantics: residual mixing stays near identity, while read/write signal and representation content concentrate in one stream.
- Using **Learned Stream Scaling (LSS)**—a minimal, near-identity diagonal parameterization at stream expansion—we show that a small controlled symmetry break reduces collapse and improves mHC variants **without changing the core HC operator**.

## Preparation

Install the required packages:

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm einops
```

### Data preparation

To prepare the datasets, enter the corresponding dataset folder and run `prepare.py`:

```sh
cd data/shakespeare_char
python prepare.py

cd ../fineweb_edu
python prepare.py

cd ../openwebtext
python prepare.py
```

Data preparation typically takes **~30 minutes** (depending on your machine and disk speed).

## Training

To train a model, run `train.py`. Use `torchrun` to enable distributed training (see the original nanoGPT project for details). Combine config files to set the dataset, model scale, and method.

### Available config files

* **Model scales**

  * S: `config/small_model.py`
  * M: `config/medium_model.py`
  * L: `config/large_model.py`

* **Methods**

  * HC: `config/with_hc.py`
  * mHC: `config/with_mhc.py`
  * mHC-lite: `config/with_mhc_lite.py`
  * Residual: (default)

* **Datasets**

  * OpenWebText: `config/train_owt.py`
  * FineWeb-Edu: `config/train_fineweb_edu.py`

### Example

Train a **small (S)** model with **mHC-lite** on **OpenWebText**:

```sh
torchrun --standalone --nproc_per_node=8 train.py \
  config/train_owt.py config/small_model.py config/with_mhc_lite.py
```

Set `--nproc_per_node` to the number of GPUs you have.

You can use `run.sh` as a starting point for batch experiment scripts.

## Analyze

Run `train_analysis.py` with `config/with_mhc_analysis.py` to run diagnostics on a checkpoint.

Training runs create output directories and checkpoints. For analysis, pass `--out_dir` to point at the run to resume from. Analysis requires checkpoints trained with mHC enabled.

### Example

To analyze a checkpoint from a **small** model trained on **OpenWebText**, set `--out_dir` to `out-owt-small-mhc`:

```sh
python train_analysis.py \
  config/train_owt.py config/with_mhc_analysis.py config/small_model.py \
  --out_dir=out-owt-small-mhc --init_from=resume
```

After the analysis run, results are saved to `log_out/infos.pkl`. Then:

```sh
python -m analyze.h_and_nu
```

generates figures under `analyze/`.

## Acknowledgements

- This codebase is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT).
- HC / mHC / mHC-lite methodology follows [mHC-lite (arXiv)](https://arxiv.org/abs/2601.05732) and related work; our Hyper-Connection layer design is informed by [hyper-connections](https://github.com/lucidrains/hyper-connections).
- We thank the [mHC reproduction](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections) project for early inspiration.
