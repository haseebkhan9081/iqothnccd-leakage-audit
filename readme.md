# iqothnccd-leakage-audit

Code and frozen artefacts for a study of **near-duplicate leakage in the
IQ-OTH/NCCD lung cancer benchmark**, and of a segmentation-then-classification
pipeline built on it.

Headline findings, each reproduced by a notebook below:

- Under the image-level partition conventional on this benchmark, **41% of the
  training set is a near-duplicate (cosine ≥ 0.98) of some test image**.
- Deleting those images from training alone, with the test folds held
  byte-identical, costs **0.076 balanced accuracy (95% CI 0.050–0.103)** beyond
  an equally sized random deletion, rising to 0.199 at a looser threshold.
- **A recovered group count matching a documented case count is almost no
  evidence that patients were recovered**: scored against real identity on MSD
  Task06_Lung, the procedure returns 65 groups against 63 true cases at 0.022
  pairwise precision.

## Notebooks

Each is self-contained and runs on a free Kaggle or Colab T4.

| notebook | what it does |
|---|---|
| `FYP_Phase0_Baseline.ipynb` | Stage-by-stage evaluation of the deployed pipeline; similarity grouping; near-duplicate counts |
| `FYP_Phase2C_Classification.ipynb` | Classification ablation under the grouped split |
| `FYP_Phase2S_Segmentation.ipynb` | U-Net retrained on MSD Task06_Lung; per-volume 3D Dice |
| `FYP_Phase2D_Detection.ipynb` | Transfer to LUNA16 subset 0; FROC and CPM |
| `FYP_Phase2X_CrossValidation.ipynb` | Five architectures × two partition protocols × five folds |
| `FYP_Phase2G_GroupingValidation.ipynb` | Scores the grouping against **known** patient identity on MSD |
| `FYP_Phase2R_RandomGroupControl.ipynb` | Random size-matched blocks: is the effect the blocking? |
| `FYP_Phase2N_DedupArm.ipynb` | Deletes near-duplicates from training only, test folds fixed |
| `FYP_Phase2T_DedupSweep.ipynb` | The same, swept across similarity thresholds |

`split_seed42.csv` is the frozen partition the IQ-OTH/NCCD notebooks read for
class labels and group membership. Note that Phase 2X draws its own five-fold
partitions from it rather than using its stored splits, and Phase 2G runs on MSD
and does not read it at all. Columns:
`file`, `label`, `y`, `group`, `split`, `split_grouped`.

## Releases

- **`weights-v1`** — original final-year-project weights (U-Net + ViT classifier)
- **`msd-unet-v2`** — U-Net retrained on MSD Task06_Lung

## A note on the name

This repository was previously called `LViT_Vision_Transformer`. That name was
inherited from the original project, which referred to its patch classifier as
"LViT" after a language-conditioned segmentation model of the same name. The
classifier here contains no language modality and no text-guided attention; it
is a conventional Vision Transformer patch classifier. The repository was
renamed so that neither system is misrepresented.

## Reproducing

Notebooks fetch the dataset from Kaggle (`hamdallak/the-iqothnccd-lung-cancer-dataset`)
and read `split_seed42.csv` from this repository, so results are comparable
across runs. Expect GPU non-determinism of a few points on individual folds;
the effects reported in the paper are stated with intervals for that reason.
