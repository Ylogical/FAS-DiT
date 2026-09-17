# FAS-DiT

This is the official implementation of the TMI paper under review:
**FAS-DiT: Frequency-Adaptive Structure-Aware Diffusion Transformer for Medical
Image Segmentation**.

FAS-DiT casts segmentation as an image-conditioned denoising process in the
**segmentation-mask space**: a Diffusion Transformer backbone denoises a noisy
mask conditioned on the medical image and predicts the clean mask $\hat{x}_0$
directly at every timestep. Two designs adapt the plain Diffusion Transformer to
medical segmentation: **MS-DSE** (Multi-Scale Dual-Stream Encoder) with its
symmetric decoder, and **DFCA** (Diffusion Frequency Cross-Attention) with a
learnable **PBDF** (Parametric Band-Decoupling Filter).

![FAS-DiT framework](assets/framework.png)

This repository contains the **training and inference code for the full model
only**. The network is fixed: there are no module switches, and training and
inference always build the same architecture.

---

## Method

At a denoising step `t`, the image `I` and the noisy mask `x_t` go through the
two **MS-DSE** branches `E_img` and `E_msk`, which produce backbone tokens plus
multi-scale skip features each. The mask tokens run through `L` FAS-DiT blocks
of timestep-modulated self-attention, and **DFCA** injects the image condition
after every block. The symmetric multi-scale decoder then upsamples the refined
tokens, fusing the image-stream and mask-stream skips at each scale, and the
output layer gives the clean-mask estimate. The image condition reaches the mask
stream through DFCA only, so image tokens never enter the backbone
self-attention.

The output layer is a `tanh` for binary segmentation and, for the multi-class
Synapse task, `x0 = 2 * softmax(z) - 1` over the `K` class channels: a signed
one-hot target satisfies `sum_c x0[c] = 2 - K` at every pixel, which the softmax
head matches by construction, so a small organ is not dragged towards `-1` by
its own negatives and left in the saturated region of a per-channel `tanh`.

![DFCA module](assets/dfca.png)

DFCA takes the mask tokens as queries and the image tokens as keys/values, and
splits the latter by frequency band into a semantic stream and a detail stream,
each with its own attention temperature. **PBDF** makes that band split
learnable through two bandwidths `sigma_s < sigma_n`, and a timestep-dependent
gate `alpha(t)` mixes the two streams, shifting from semantics at high noise to
detail as `t` decreases.

<p align="center">
  <img src="assets/pbdf.png" width="62%" />
</p>

---

## Installation

```bash
conda env create -f environment.yaml
conda activate fas-dit
```

or, with an existing PyTorch installation:

```bash
pip install -r requirements.txt
```

## Data

No data is shipped with this repository. Download the datasets from their
original sources, then arrange them **yourself** into the layout below — the
loader reads nothing else.

| Dataset | Task | Size | Input | Source |
| --- | --- | --- | --- | --- |
| GlaS | Gland segmentation (H&E) | 165 images | 256×256 | https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/ |
| MoNuSeg | Nuclei segmentation (H&E) | 51 images | 512×512 | https://monuseg.grand-challenge.org/Data/ |
| PH2 | Skin lesion (dermoscopy) | 200 images | 256×256 | https://www.fc.up.pt/addi/ph2%20database.html |
| IMID | Islet tissue (H&E), private | 400 images | 256×256 | not public |
| TNBC | Nuclei segmentation (H&E) | ~50 images | 512×512 | https://zenodo.org/records/1174343 |
| Synapse | Multi-organ abdominal CT, 8 organs | 30 volumes | 224×224 | https://www.synapse.org/#!Synapse:syn3193805/wiki/217789 |

TNBC is only the target domain of the cross-dataset experiment (a MoNuSeg model
evaluated on TNBC without fine-tuning), so it needs a `test/` split only.

Synapse is the one **multi-class** dataset (background + 8 organs = 9 classes)
and keeps the preprocessed TransUNet layout rather than the `processed_*` one;
see its own subsection below.

### Directory layout (binary datasets)

Put the prepared folders in the project root, one per dataset:

```
FAS-DiT/
├── processed_glas/
│   ├── train/
│   │   ├── images/    img_001.png, img_002.png, ...
│   │   └── masks/     img_001.png, img_002.png, ...
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── processed_monuseg/   same structure
├── processed_ph2/       same structure
├── processed_imid/      same structure
└── processed_tnbc/      test/ only (cross-dataset evaluation)
```

Rules the loader relies on:

* An image and its mask are matched **by file name stem**, so `img_001.png` in
  `images/` pairs with `img_001.png` (or `img_001.tif`, ...) in `masks/`.
  Accepted extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.
* Images are RGB (grayscale is converted); masks are **single-channel binary**,
  background 0 and foreground 255 (anything > 127 counts as foreground).
* Images and masks may be stored at any size — they are resized to
  `--image_size` on load (bilinear for images, nearest for masks) — but storing
  them already at the target size keeps loading fast.
* The three splits are fixed on disk. The paper uses a sample-wise (patient-wise
  for GlaS) 6 : 2 : 2 train / val / test split; MoNuSeg instance annotations are
  merged into a binary foreground mask.

### Synapse

Use the preprocessed Synapse release of the TransUNet authors (the `train_npz` /
`test_vol_h5` archives; slices are windowed to the `[-125, 275]` HU range and
stored at 512×512). Put it in the project root as:

```
FAS-DiT/
└── Synapse/
    ├── train_npz/caseXXXX_sliceYYY.npz   2D slices, image + label, 18 cases
    ├── test_vol_h5/caseXXXX.npy.h5       3D volumes, 12 cases
    └── split.json                        written by prepare_synapse_split.py
```

Then write the split file once:

```bash
python prepare_synapse_split.py            # 18 train / 12 test, no val (the paper)
python prepare_synapse_split.py --val_cases 4    # hold 4 cases out for early stopping
```

The split is defined at the **case** level, so slices of one patient never
straddle two splits. Labels are `0` (background) and `1..8` for aorta,
gallbladder, left kidney, right kidney, liver, pancreas, spleen and stomach; the
loader turns them into 9 signed one-hot channels in `{-1, +1}` and also returns
the integer label map for the cross-entropy term. Training and inference run at
224×224, and the test slices additionally carry their native 512×512 label so
the evaluation can score the restored prediction (see *Inference*).

Synapse needs `h5py` for the test volumes; it is in `requirements.txt` and
`environment.yaml`.

## Training

Image size, model variant, epochs, learning rate, batch size, dropout and early
stopping are all selected from `--dataset`; see `_DATASET_DEFAULTS` in
`train.py`.

```bash
python train.py --dataset glas
python train.py --dataset ph2
python train.py --dataset imid
python train.py --dataset monuseg          # 512x512, patch size 32
python train.py --dataset synapse          # 224x224, 9 classes, multi-class

# resume
python train.py --dataset glas --resume checkpoints/<run>/checkpoint_epoch100.pth
```

| Dataset | Input | Variant | Batch | LR | Weight decay | λ_CE |
| --- | --- | --- | --- | --- | --- | --- |
| GlaS | 256² | FAS-DiT-B/16 | 8 | 2e-4 | 0.05 | 0 |
| PH2 | 256² | FAS-DiT-B/16 | 8 | 2e-4 | 0.05 | 0 |
| IMID | 256² | FAS-DiT-B/16 | 4 | 2e-4 | 0.05 | 0 |
| MoNuSeg | 512² | FAS-DiT-B/32 | 4 | 1e-4 | 0.05 | 0 |
| Synapse | 224² | FAS-DiT-B/16 | 8 | 2e-4 | 0.02 | 0.5 |

Common to all runs: `T = 200` diffusion steps, cosine noise schedule, direct
$\hat{x}_0$ prediction, AdamW, MSE + soft Dice loss (`λ_Dice = 1.0`), EMA
(decay 0.9999), cosine LR schedule with warmup, and training from scratch (no
pretrained weights). On Synapse a weighted cross-entropy term (`λ_CE = 0.5`) is
added on top; it is defined only under the softmax x0 head, so `--ce_weight` is
rejected on the binary datasets.

The binary runs are capped at 20k epochs and stopped early once the validation
mIoU stops improving (not before 10k epochs), which is where the reported models
land; evaluate `best_model.pth` (best EMA validation mIoU).

**Synapse has no validation split** under the standard 18 / 12 protocol, so
early stopping and best-model selection switch themselves off: the run goes the
full `--epochs` (default 10000) and `final_model.pth` is the model to evaluate.
Add `--save_freq N` for periodic checkpoints on a long run, or build the split
with `--val_cases 4` if you would rather have early stopping back.

Checkpoints go to `checkpoints/<dataset>_<model>_<timestamp>/`, TensorBoard logs
and loss curves to `logs/`.

Variants: `FAS-DiT-{S,B,L,H}/{16,32}`, i.e. depth 4 / 6 / 8 / 12 with hidden
size 512 / 768 / 1024 / 1280 and patch size 16 or 32. The paper uses `B`.

## Inference

```bash
python inference.py \
    --checkpoint checkpoints/<run>/best_model.pth \
    --dataset glas --use_ema --K 25 --save_images
```

Reports mIoU, DSC, Sensitivity, Accuracy, Precision, HD95, GED and ECE, plus
the parameter count and GFLOPs; results are written to
`results/<dataset>/metrics.txt`, and `--save_images` additionally writes the
predictions and image/GT/prediction comparison strips.

`--K` is the number of independent DDIM trajectories averaged into the MMSE
estimate (`--K 1` is fastest, `--K 25` is the setting used in the paper);
`--ddim_steps` sets `T'` (20 by default) and `--eta 0` keeps sampling
deterministic. The model variant, input size and number of mask channels are
read from the checkpoint, so they do not have to be repeated.

Synapse is evaluated per **volume**, which is what the compared methods report:

```bash
python inference.py --checkpoint checkpoints/<synapse-run>/final_model.pth \
    --dataset synapse --use_ema --K 25
```

Each slice is predicted by argmax over the 9 class channels, restored from
224×224 to the native 512×512 label resolution by nearest-neighbour
upsampling, and the slices of a case are stacked back into a volume. Every organ
is scored on that volume, averaged over the 12 cases first and then over the 8
organs (the TransUNet aggregation order). The report gives this case-wise 3D DSC
and HD95 per organ and in total, plus the slice-level scores for reference. A
`(case, organ)` pair that is empty on both sides counts as DSC 1; HD95 is
undefined whenever either side is empty and counts as 0 there, with the number
of such pairs printed next to it. HD95 is in voxels, at unit spacing.

Cross-dataset generalization (a MoNuSeg model evaluated on TNBC, no
fine-tuning):

```bash
python inference.py --checkpoint checkpoints/<monuseg-run>/best_model.pth \
    --dataset tnbc --use_ema --K 25
```

## Repository layout

```
model_fas_dit.py          model definition (MS-DSE, DFCA/PBDF, DiT backbone) and variants
diffusion_utils.py        cosine schedule, DDIM sampling, MSE + Dice (+ CE) loss
dataset.py                binary and Synapse loaders, dataloader factories
prepare_synapse_split.py  writes Synapse/split.json (case-level split)
train.py                  training entry point
inference.py              evaluation and metrics (binary and multi-class)
training_logger.py        logging and loss curves
util/model_util.py        2D RoPE, sincos positional embedding, RMSNorm
```



## License

Released under the [MIT License](LICENSE).
