# Facial Recognition with Eigenfaces (PCA / SVD)

Final project for Approximate Models / Scientific Computing (UFRJ) — a simplified
facial recognition implementation using the classic **Eigenfaces** technique
(Turk & Pentland, 1991), applying **PCA** and **SVD decomposition** built from
linear algebra fundamentals: eigenvalues, eigenvectors, orthogonality, and
low-rank approximation.

## Overview

Each face (a 64×64 pixel image) is treated as a vector of 4096 numbers. Given a
set of 400 faces, PCA finds the directions of greatest variation among them —
the **eigenfaces** — allowing each face to be compressed to a few dozen
coefficients without losing the information relevant for identification.
Identifying a new person is done by comparing, in this reduced space, the
distance between the test face and all known training faces.

## Project highlights

- PCA implemented **from scratch**, via eigendecomposition of $A^TA$ (does not
  use the ready-made `sklearn.decomposition.PCA`) — with the math commented in
  the code.
- Comparison against a **direct SVD** as a sanity check, including condition
  number analysis and a discussion of numerical stability.
- Nearest-neighbor classification with proper train/test validation (no data
  leakage).
- Confusion matrix and a step-by-step walkthrough of an individual
  classification.
- Single narrative notebook, ready to run on Google Colab or locally.

## Dataset

**Olivetti Faces** — 40 people, 10 photos each (400 images), 64×64 grayscale
pixels. Loaded via `sklearn.datasets.fetch_olivetti_faces()`.

> **Note:** downloading this dataset (hosted on Figshare) can intermittently
> fail with a `403 Forbidden` error in cloud environments such as Google Colab —
> this is a known, recurring issue on Figshare's side, not a problem with this
> project's code. See the [Troubleshooting](#troubleshooting) section below.

## Repository structure

```
eigenfaces/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                           # cached dataset (git-ignored)
├── src/
│   ├── load_data.py                # loads the Olivetti dataset into matrix A
│   ├── pca.py                      # PCA via eigenvalues of AᵀA, projection, reconstruction
│   ├── eigenfaces.py               # eigenfaces, reconstruction, explained variance
│   ├── recognize.py                # nearest-neighbor classification
│   ├── confusion_matrix.py         # classification confusion matrix
│   └── compare_svd.py              # sanity check: PCA via AᵀA vs. direct SVD
├── notebooks/
│   └── eigenfaces_complete.ipynb   # consolidated narrative notebook (recommended)
├── results/                        # generated plots and images
└── report/                         # full technical project report
```

## How to run

### Option 1 — Notebook (recommended)

Open `notebooks/eigenfaces_complete.ipynb` locally (Jupyter/VS Code) or upload it
to [Google Colab](https://colab.research.google.com). The notebook is
self-contained — it doesn't depend on the files in `src/`, and runs start to
finish on its own.

### Option 2 — Individual scripts

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd src
python load_data.py             # tests dataset loading
python eigenfaces.py            # eigenfaces, reconstruction, explained variance
python recognize.py             # classification and accuracy vs. number of components
python confusion_matrix.py      # confusion matrix
python compare_svd.py           # PCA vs. direct SVD comparison
```

> On Arch Linux (and other distros with a system-managed Python), use a venv as
> above, or add `--break-system-packages` to `pip install` if you'd rather
> install globally.

## Results

| Metric | Value |
|---|---|
| Components for 95% explained variance | 38 |
| Components for 99% explained variance | 177 |
| Classification accuracy (50 components) | 94% |
| Confusion matrix errors | 6 out of 100 test photos |
| Numerical difference, PCA (via AᵀA) vs. direct SVD | ~10⁻⁶, consistent with theory |

Full details — including the mathematical background, bugs found during
development and how they were fixed, and answers to the most likely questions
from an evaluation committee — are in [`report/`](./report).

## Known limitations

- The classifier always assigns the test photo to one of the 40 people in the
  training set — there is no "unknown person" rejection.
- Assumes faces are already centered and standardized (as in the Olivetti
  dataset); real-world photos would require additional preprocessing (face
  detection and alignment).

## Troubleshooting

**`HTTPError: 403 Forbidden` when running `fetch_olivetti_faces()` on Colab:**
this is an intermittent block by Figshare against requests coming from
datacenters/cloud environments, not an issue with this code. Solutions, in
order of preference:
1. Retry after a few minutes (usually temporary).
2. Run locally, where the dataset may already be cached
   (`~/scikit_learn_data/`).
3. Export the already-downloaded local dataset to a `.npz` file
   (`np.savez_compressed("olivetti_faces.npz", images=data.images, targets=data.target)`),
   manually upload that file to Colab, and load it via `np.load()` instead of
   `fetch_olivetti_faces()` — this removes the external network dependency.

## References

- Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of
  Cognitive Neuroscience.
- Dataset: [Olivetti Faces (AT&T)](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset)

## Author

Daniel — Computer Science, Instituto de Computação, UFRJ
