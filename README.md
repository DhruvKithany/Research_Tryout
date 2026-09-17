# Geometry of Affect: Decoding and Manifold Analysis of 27 Fine-Grained Emotions from Human EEG

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-modular-green.svg)](https://github.com/)
[![Tests](https://img.shields.io/badge/tests-15%20passed-success.svg)](tests/)

Author: Dhruv Kithany  
Institution: University of Wisconsin–Madison  
Application: Exploratory Research Exercise — Computational Neuroengineering, Machine Learning & Geometric Data Analysis (Bhaskar Lab)  
Dataset: [EEGEmotions-27 Dataset](https://github.com/huytungst/EEGEmotions-27) (*IEEE Access*, 2025)

---

## Quick Summary/Overview

Human emotional experience is notoriously complex. While classical affective psychology has debated whether emotions are discrete categories (Ekman's 6 basic emotions) or continuous low-dimensional axes (Russell's Valence-Arousal circumplex), recent behavioral discoveries by Cowen & Keltner (*PNAS*, 2017) demonstrated that emotion is best represented as a continuous, high-dimensional manifold bridged by semantic gradients across 27 distinct varieties. 

This research project investigates a central question in computational neuroengineering:  
> Does the geometric and topological manifold of human scalp electroencephalography (EEG) reflect the high-dimensional structure of emotional experience, and can fine-grained affective states be reliably decoded across unseen human subjects?

Using the newly released EEGEmotions-27 dataset (88 participants, 14-channel 256Hz EEG recorded during evocative video elicitation), this repository implements a comprehensive, end-to-end computational neuroengineering framework spanning neurobiological topography, manifold learning & representational similarity analysis (RSA), rigorous subject-independent decoding benchmarks, and class-conditional generative modeling.

```
                       +-------------------------------------------------------------+
                       |              EEGEmotions-27 Raw Data (88 Subjects)           |
                       +-------------------------------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   Feature Extraction (33 x 14 = 462)   |
                                  |  - Spectral Powers (Delta..Gamma, RBA)|
                                  |  - Entropies (Shannon, Renyi, Tsallis)|
                                  |  - Non-linear Dynamics & Hjorth Mob.  |
                                  +---------------------------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
                    v                                 v                                 v
   +---------------------------------+  +----------------------------+  +-------------------------------+
   |   1. Neurobiology & Topography  |  | 2. Manifold & Topology     |  | 3. Subject-Independent        |
   | - 10-20 Scalp Interpolation     |  | - Diffusion / Laplacian    |  |    Decoding Benchmark         |
   | - Frontal Alpha Asymmetry (FAA) |  |   Eigenmap Manifolds       |  | - 4-Class Valence/Arousal     |
   | - Multi-Band Spatial Power      |  | - RSA vs Psychological     |  |   (56.6% Acc vs 25% Chance)   |
   | - Time-Series & Welch PSD       |  |   Circumplex (p = 0.007)   |  | - 27-Class Fine-Grained       |
   +---------------------------------+  | - Two-NN Intrinsic Dim     |  |   (30.8% Top-1, 79.8% Top-5)  |
                                        | - Hierarchical Ward Tree   |  +-------------------------------+
                                        +----------------------------+                  |
                                                      |                                 v
                                                      v                 +-------------------------------+
                                        +----------------------------+  | 4. Conditional EEG Generator  |
                                        |  Cortical & Spectral       |  | - Ledoit-Wolf Covariance      |
                                        |  Feature Importance        |  | - Synthetic Sample Synthesis  |
                                        +----------------------------+  | - Fréchet Distance Evaluation |
                                                                        +-------------------------------+
```

---

## Key Findings

1. Neural Geometry Correlates with Psychological Affective Space (p = 0.007):
   - Using Representational Similarity Analysis (RSA), we computed pairwise centroid distances between all 27 emotion classes in the 462-dimensional neural feature space and compared them against normative Valence-Arousal coordinates.
   - Upper-triangular distance correlation revealed a statistically significant alignment (r = 0.1420, p = 7.70e-3; Spearman rho = 0.1431, p = 7.25e-3), demonstrating that the physical brain manifold preserves topological proximity of human emotional experiences.

2. Intrinsic Manifold Dimensionality:
   - Applying the Two-NN intrinsic dimension estimator (Facco et al., *Sci. Rep.* 2017) revealed that the effective intrinsic dimensionality of the human EEG affective manifold is d ≈ 13.4.
   - PCA cumulative variance shows that 5 orthogonal components explain 80% of feature variance, 9 explain 90%, and 14 components explain 95%—closely reflecting the 14 cortical electrode channels.

3. Frontal Alpha Asymmetry (FAA) Reflects Motivational Direction:
   - Quantified prefrontal hemispheric lateralization: FAA = ln(Alpha_AF4) - ln(Alpha_AF3)
   - Positive FAA (relative left prefrontal dominance, indexing approach motivation and appetitive drive) characterizes emotions like *Joy*, *Sexual Desire*, *Excitement*, and *Craving*.
   - Negative FAA (relative right prefrontal dominance, indexing withdrawal motivation and behavioral inhibition) characterizes emotions like *Sadness*, *Fear*, *Horror*, and *Disgust*.

4. Decoding Benchmark on Strict Subject-Independent Splits (Zero Data Leakage):
   - Evaluated under strict Leave-Subjects-Out (Subject-Independent Group Split) across 74 training subjects and 14 held-out test subjects:
     - 4-Class Valence-Arousal Quadrant Decoding: 56.64% accuracy (chance: 25.0%, Top-3: 98.12%).
     - 27-Class Fine-Grained Emotion Decoding: 30.83% Top-1 accuracy, 65.13% Top-3 accuracy, and 79.83% Top-5 accuracy! This represents an 8.3-fold increase over the 3.70% random chance baseline.

5. Class-Conditional Synthetic EEG Generation:
   - Implemented an analytical generative model with Ledoit-Wolf covariance shrinkage that models p(X | y = c) = N(mu_c, Sigma_c), enabling synthetic signal synthesis with low Fréchet distances (FD = 116.88 for Sadness, FD = 401.60 for Anger).

---

## Subject-Independent Decoding Benchmark

All models were evaluated under strict subject-independent splitting (`GroupShuffleSplit`, random state 42) ensuring that no subject's time-series or segments appear in both training and testing.

### Task A: 4-Class Valence-Arousal Quadrants (Chance = 25.0%)
| Model | Test Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Cohen's Kappa | Top-3 Acc | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 56.64% | 44.86% | 0.4340 | 0.5446 | 0.3639 | 98.12% | 5.97s |
| Extra Trees | 53.79% | 42.58% | 0.4094 | 0.5162 | 0.3239 | 95.67% | 0.61s |
| Deep MLP Neural Net | 50.20% | 42.98% | 0.4315 | 0.4909 | 0.2793 | 96.70% | 20.45s |
| Logistic Regression (L2)| 42.11% | 33.54% | 0.3144 | 0.3910 | 0.1464 | 91.34% | 6.31s |
| Ridge Classifier | 41.94% | 33.37% | 0.3149 | 0.3921 | 0.1442 | 90.37% | 0.17s |
| Nearest Centroid (Cosine)| 27.58% | 31.19% | 0.2599 | 0.2936 | 0.0640 | 81.42% | 0.10s |

### Task B: 27-Class Fine-Grained Cowen Emotions (Chance = 3.70%)
| Model | Top-1 Acc | Top-3 Acc | Top-5 Acc | Balanced Acc | Macro F1 | Cohen's Kappa | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 30.83% | 65.13% | 79.83% | 30.84% | 0.3028 | 0.2817 | 8.80s |
| Deep MLP Neural Net | 27.92% | 61.99% | 80.28% | 27.82% | 0.2740 | 0.2513 | 54.47s |
| Extra Trees | 26.04% | 56.35% | 74.36% | 26.05% | 0.2520 | 0.2320 | 1.05s |
| Logistic Regression (L2)| 12.25% | 30.14% | 47.98% | 12.33% | 0.1043 | 0.0889 | 17.48s |
| Ridge Classifier | 10.48% | 28.03% | 41.71% | 10.62% | 0.0844 | 0.0706 | 0.16s |
| Nearest Centroid (Cosine)| 5.36% | 16.87% | 26.89% | 5.35% | 0.0382 | 0.0172 | 0.07s |

---

## Neurobiological & Geometric Figures

### 1. Frontal Alpha Asymmetry (FAA) Across 27 Emotional States
Hemispheric lateralization at prefrontal sites (AF4 - AF3) ranks emotions along the approach-withdrawal axis:
![Frontal Alpha Asymmetry](figures/01_frontal_alpha_asymmetry.png)

### 2. Multi-Band Scalp Topographies (10-20 System)
Interpolated 2D scalp power maps across Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), and Gamma (30-45 Hz):
* Joy (High Valence, High Arousal):
  ![Topomap Joy](figures/01_topomap_multiband_joy_hvha.png)
* Calmness (High Valence, Low Arousal):
  ![Topomap Calmness](figures/01_topomap_multiband_calmness_hvla.png)
* Anger (Low Valence, High Arousal):
  ![Topomap Anger](figures/01_topomap_multiband_anger_lvha.png)
* Sadness (Low Valence, Low Arousal):
  ![Topomap Sadness](figures/01_topomap_multiband_sadness_lvla.png)

### 3. Manifold Learning & Representational Geometry
* Diffusion / Laplacian Eigenmaps:
  Low-dimensional non-linear diffusion coordinates revealing the grouping of affective states:
  ![Diffusion Manifold](figures/02_manifold_diffusion_quadrants.png)

* Pairwise Neural Distance Matrix (27 Emotions):
  Euclidean distance heatmap between emotion centroids in 462-dimensional space:
  ![Distance Matrix](figures/02_neural_emotion_distance_matrix.png)

* Hierarchical Ward Dendrogram:
  Brain state hierarchical taxonomy illustrating which emotional states share neural activation patterns:
  ![Hierarchical Dendrogram](figures/02_emotion_hierarchical_dendrogram.png)

### 4. Decoding Confusion Matrices
* Valence-Arousal Quadrants:
  ![Confusion Matrix Quadrants](figures/03_confusion_matrix_quadrants.png)
* 27 Fine-Grained Emotions:
  ![Confusion Matrix 27 Emotions](figures/03_confusion_matrix_cowen27.png)

### 5. Neuro-Feature Importance
Gini impurity reduction partitioned by spectral band and anatomical cortical lobe:
![Feature Importance](figures/03_feature_importance_bands_lobes.png)

### 6. Class-Conditional Synthetic Generation
PCA projection showing overlap between held-out empirical EEG test data and synthetic feature vectors generated via Ledoit-Wolf covariance shrinkage:
![Generative Overlap](figures/04_generative_feature_fidelity.png)

---

## Repo Structure

```
Research_Tryout/
├── .gitignore                          # Clean git ignore (excludes 3.4GB raw recordings & caches)
├── LICENSE                             # MIT License
├── README.md                           # Comprehensive documentation and research report
├── requirements.txt                    # Pinned Python dependencies
├── pyproject.toml                      # Modern packaging configuration
├── setup.py                            # Package setup script
│
├── eeg_affect/                         # Core Python Package
│   ├── __init__.py                     # Package entry point
│   ├── config.py                       # Electrode positions, Cowen dictionary, circumplex coordinates
│   ├── data/                           # Data loading and cross-subject partitioning
│   │   ├── __init__.py
│   │   ├── loader.py                   # Loads features, metadata, and raw time-series
│   │   ├── split.py                    # SubjectGroupSplitter (Leave-Subjects-Out)
│   │   └── preprocessor.py             # Robust/Standard scaling, variance filtering
│   ├── features/                       # Spectral and neurobiological feature engineering
│   │   ├── __init__.py
│   │   ├── asymmetry.py                # Frontal Alpha Asymmetry (FAA) & bilateral metrics
│   │   ├── bands.py                    # Band power extraction & cortical lobe grouping
│   │   └── statistics.py               # Time-series Hjorth parameters, Shannon entropy, PSD
│   ├── geometry/                       # Geometric & Topological Manifold Analysis
│   │   ├── __init__.py
│   │   ├── manifold.py                 # PCA, Diffusion / Laplacian Eigenmaps, t-SNE, MDS
│   │   ├── distance.py                 # Centroid distance matrices, RSA Mantel test, Ward tree
│   │   └── intrinsic_dim.py            # Two-NN and PCA variance intrinsic dimension estimation
│   ├── models/                         # Decoding & Generative Architectures
│   │   ├── __init__.py
│   │   ├── baselines.py                # Ridge, LogisticRegression, CosineNearestCentroid
│   │   ├── ensembles.py                # RandomForest, ExtraTrees, HistGradientBoosting
│   │   ├── neural.py                   # Deep Multi-Layer Perceptron (MLP)
│   │   └── generator.py                # Class-Conditional Generator with Ledoit-Wolf shrinkage
│   ├── evaluation/                     # Evaluation metrics and benchmark suites
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Top-1/3/5 accuracy, Macro F1, Balanced Acc, Cohen's Kappa
│   │   └── benchmark.py                # Cross-subject benchmarking engine
│   └── visualization/                  # Publication-grade plotting modules
│       ├── __init__.py
│       ├── topomap.py                  # 10-20 Scalp Topography interpolation & renderer
│       ├── manifold_plots.py           # 2D/3D Manifold projections & distance heatmaps
│       ├── neuro_plots.py              # Frontal Alpha Asymmetry & regional bar plots
│       └── confusion.py                # Confusion matrices & hierarchical dendrograms
│
├── scripts/                            # Executable CLI Pipelines
│   ├── 01_visualize_neurobiology.py    # Generates topomaps, FAA, and raw PSD traces
│   ├── 02_analyze_manifold_geometry.py # Runs manifold embedding, RSA, and intrinsic dim
│   ├── 03_run_decoding_benchmark.py    # Runs subject-independent benchmarks & feature importance
│   ├── 04_generate_synthetic_eeg.py    # Fits conditional generator & evaluates Fréchet distance
│   └── run_all.py                      # One-click master reproduction script
│
├── tests/                              # Comprehensive Automated Test Suite
│   ├── __init__.py
│   ├── run_tests.py                    # Test runner (runs without external test dependencies)
│   ├── test_data_loader.py             # Tests data loading, shapes, and group splits
│   ├── test_features.py                # Tests FAA, band grouping, and Hjorth statistics
│   ├── test_geometry.py                # Tests manifold methods, distances, and Two-NN
│   └── test_models.py                  # Tests baseline, ensemble, neural, and generator models
│
├── notebooks/                          # Interactive Exploration
│   └── 01_exploratory_eeg_affect_walkthrough.ipynb
│
├── figures/                            # High-resolution (300 DPI) publication figures & metrics
└── data/                               # Clean, lightweight data directory
    ├── emotivX_channels_location.ced   # 14 electrode coordinates (Emotiv 10-20)
    ├── participants_info.csv           # Demographic metadata (Age, Gender, Nation)
    ├── eeg_features_extracted.csv      # Extracted feature matrix (462 features)
    └── sample_raw/                     # Curated raw recordings for instant time-series testing
```

---

## Quick Start and Reproduction Instructions

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/EEG-Affect-Geometry.git
cd EEG-Affect-Geometry

# Install dependencies
pip install -r requirements.txt
# Or install in editable mode
pip install -e .
```

### 2. Run All Experiments (One-Click Reproduction)
Execute the complete four-stage pipeline end-to-end:
```bash
python scripts/run_all.py
```
This runs:
1. `scripts/01_visualize_neurobiology.py`: Produces 10-20 scalp topomaps, Frontal Alpha Asymmetry bar plots, and raw time-series/PSD plots.
2. `scripts/02_analyze_manifold_geometry.py`: Computes 2D Diffusion/PCA embeddings, pairwise neural distance matrix, RSA test against psychological circumplex, Ward dendrogram, and Two-NN intrinsic dimensionality.
3. `scripts/03_run_decoding_benchmark.py`: Runs cross-subject evaluation across 6 model paradigms on both 4-class quadrants and 27 Cowen emotions; computes top-k accuracies and feature importances.
4. `scripts/04_generate_synthetic_eeg.py`: Fits class-conditional multivariate models with Ledoit-Wolf shrinkage, generates synthetic feature vectors, and computes Fréchet distance fidelity.

### 3. Run Automated Unit Tests
Verify all components with zero external test runners:
```bash
python tests/run_tests.py
```
*(All 15 unit tests across data loaders, feature engineering, manifold algorithms, and models pass.)*

### 4. Interactive Jupyter Notebook
Launch the interactive tutorial walkthrough:
```bash
jupyter notebook notebooks/01_exploratory_eeg_affect_walkthrough.ipynb
```


## References

1. Dataset Paper: Phuong, H.-T., Im, E.-T., Oh, M.-S., & Gim, G.-Y. (2025). *EEGEmotions-27: A Large-Scale EEG Dataset Annotated With 27 Fine-Grained Emotion Labels*. IEEE Access, 13, 176915–176932. [DOI: 10.1109/ACCESS.2025.3620677](https://doi.org/10.1109/ACCESS.2025.3620677)
2. Review Paper: Phuong, H.-T., Im, E.-T., Oh, M.-S., & Gim, G.-Y. (2025). *EEG-Based Emotion Recognition: A Review and Emerging Paths*. IEEE Access, 13, 165037–165060. [DOI: 10.1109/ACCESS.2025.3610918](https://doi.org/10.1109/ACCESS.2025.3610918)
3. Affective Geometry: Cowen, A. S., & Keltner, D. (2017). *Self-report captures 27 distinct categories of emotion bridged by continuous gradients*. Proceedings of the National Academy of Sciences (PNAS), 114(38), E7900–E7909.
4. Frontal Asymmetry: Davidson, R. J. (1992). *Anterior cerebral asymmetry and the nature of emotion*. Brain and Cognition, 20(1), 125–151.
5. Intrinsic Dimensionality: Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017). *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*. Scientific Reports, 7(1), 12140.

---

## Github License
This project is released under the [MIT License](LICENSE).
The underlying EEGEmotions-27 dataset is distributed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
