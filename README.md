# Retinal OCT analysis with a domain-adapted DINOv2 backbone

Three-stage pipeline: data preparation, continual DINOv2 self-supervised
pre-training on OCT scans, and supervised fine-tuning of the resulting
ViT-S/14 backbone on four downstream tasks. A Streamlit app demonstrates the
trained models on single B-scans.

## Layout

- `PREPROCESS_DATA/`, `prepare_dataset.py` - stage 1, data preparation
- `analyse_pretrain/dinov2_modify/`        - stage 2 integration with the official DINOv2 code
- `analyse_pretrain/method_*.py`           - backbone probes (kNN, linear probe, UMAP, patch-PCA, CLS-patch similarity)
- `finetune_octdl/`, `finetune_mmrdr/`, `finetune_corina/`, `finetune_oct5k/`
                                           - stage 3, one package per task (dataset/model/train + analyse_*)
- `retino_app.py`                          - Streamlit demo
- `early_experiments_ch4/`                 - chapter-4 pilots, kept for reference; not part of the final pipeline

## Environment

The fine-tuning, evaluation, and demo code run against a standard PyTorch stack:

    python -m venv venv
    venv\Scripts\activate            # Linux/WSL: source venv/bin/activate
    pip install -r requirements.txt

The stage-2 SSL run was carried out separately, inside the official DINOv2
repository's own conda environment (Python 3.10, CUDA) on a GPU workstation.

## Data and checkpoints

The four labeled datasets are downloaded from their public release pages
(OCTDL, MMRDR, OCT5k, Suciu et al.). The institutional pre-training collection
is not public. Datasets, trained checkpoints, and training logs are not
included in this archive because of their size; the scripts below regenerate
everything except the raw data and the SSL corpus.

## Paths to set before running

Most scripts take their paths on the command line (`--data_path`, `--csv`,
`--image_root`, `--checkpoint`); the paths printed in each script's `Usage`
docstring are examples, replace them with your own. Two paths are instead set
in files and must be edited by hand before a stage-2 run:

- `analyse_pretrain/dinov2_modify/loaders.py`, line 73 - the OCT SSL corpus
  root. Point it at the `train/` directory produced by `prepare_dataset.py`.
- `analyse_pretrain/dinov2_modify/oct_continual.yaml`, `train.output_dir` -
  where SSL checkpoints and logs are written.

## Stage 1 - data preparation

    python PREPROCESS_DATA/preprocess_data.py --help    # clean OCTDL, write the metadata csv
    python prepare_dataset.py --help                    # assemble the unlabeled SSL corpus (symlinks into train/all_scans/)

## Stage 2 - continual self-supervised pre-training

Runs on the official Meta DINOv2 code; this repo only adds the OCT integration
files under `analyse_pretrain/dinov2_modify/`. To reproduce:

1. clone the official repository:

       git clone https://github.com/facebookresearch/dinov2

2. copy the integration files into the clone (they replace the upstream
   versions of the data and training layer, adding `OCTDataset`):

       oct_dataset.py      -> dinov2/data/datasets/oct_dataset.py
       __init__.py         -> dinov2/data/datasets/__init__.py
       loaders.py          -> dinov2/data/loaders.py
       train.py            -> dinov2/train/train.py
       fsdp/__init__.py    -> dinov2/fsdp/__init__.py
       oct_continual.yaml  -> dinov2/configs/train/oct_continual.yaml

3. set the OCT corpus root in `loaders.py` and `output_dir` in
   `oct_continual.yaml` (see "Paths to set" above).

4. point the ImageNet-weights environment variable at the public ViT-S/14
   checkpoint, then launch training with the OCT config:

       export DINOV2_PRETRAIN_WEIGHTS=/path/to/dinov2_vits14_pretrain.pth
       python dinov2/train/train.py --config-file dinov2/configs/train/oct_continual.yaml

The run emits teacher checkpoints; downstream code reads the
`teacher.backbone.*` tensors.

## Stage 3 - supervised fine-tuning

One package per task. Each trains from the stage-2 checkpoint (domain-adapted)
or, with no checkpoint given, from the torch.hub ImageNet weights (baseline):

    python finetune_octdl/train.py --help

Run each package's `analyse_*.py` for metrics, confusion matrices, Grad-CAM,
and the embedding projections. The `method_*.py` scripts under
`analyse_pretrain/` probe a bare backbone (kNN, linear probe, UMAP, patch-PCA,
similarity) without fine-tuning.

## Demo

    streamlit run retino_app.py

The `sample_images/` folder ships with this source archive, so the demo runs on
example scans out of the box. Before launching, download the separately provided
artifacts from the shared drive (see "Downloading checkpoints and figures" below)
and place them at the repository root:

  - `checkpoints/`  — the SSL backbone (`model_final.rank_0.pth`) plus, per task,
                      a domain-adapted (`*_da.pth`) and an ImageNet-baseline
                      (`*_in.pth`) checkpoint (9 files total).
  - `finetune_*/results/` and `results/umap/` — the pre-generated figures shown
                      in the Reports and Latent space tabs.

## Downloading checkpoints and figures

The trained checkpoints and the generated figure folders are not committed to
this archive (they are large binary/generated artifacts). Download them here:

    <PASTE GOOGLE DRIVE LINK>

Unzip at the repository root so that `checkpoints/`, `finetune_*/results/`, and
`results/umap/` sit next to `retino_app.py`.