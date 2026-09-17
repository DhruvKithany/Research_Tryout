"""Script to generate the interactive walkthrough Jupyter Notebook using camelCase API."""

from pathlib import Path
import nbformat as nbf

projectRoot = Path(__file__).resolve().parent.parent
nbPath = projectRoot / "notebooks" / "01_exploratory_eeg_affect_walkthrough.ipynb"
nbPath.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""# Geometry of Affect: Decoding and Manifold Analysis of 27 Fine-Grained Emotions from Human EEG
### Exploratory Interactive Walkthrough
**Author:** Dhruv Kithany  
**Dataset:** EEGEmotions-27 (Phuong et al., IEEE Access 2025)  
**Laboratory Focus:** Neuroengineering, Machine Learning, Topological & Manifold Data Analysis

---

### Overview
This notebook interactively demonstrates the core components of the project:
1. **Neurobiological Topography & Frontal Alpha Asymmetry (FAA)**
2. **Representational Similarity Analysis (RSA) & Manifold Diffusion Maps**
3. **Cross-Subject (Leave-Subject-Out) Emotion Decoding Benchmark**
4. **Class-Conditional Synthetic EEG Feature Generation**
"""))

nb.cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

projectRoot = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
sys.path.insert(0, str(projectRoot))

from eeg_affect.config import CHANNELS_14, COWEN_27_EMOTIONS, QUADRANT_NAMES
from eeg_affect.data.loader import loadEegFeatures, loadChannelLocations, loadRawEeg
from eeg_affect.data.preprocessor import EEGPreprocessor
from eeg_affect.data.split import SubjectGroupSplitter
from eeg_affect.features.asymmetry import computeFrontalAlphaAsymmetry
from eeg_affect.geometry.manifold import EmotionManifoldEmbedder
from eeg_affect.geometry.distance import computeEmotionCentroids, computePairwiseGeometry, compareEegVsPsychologyGeometry
from eeg_affect.visualization.topomap import plotTopomap, plotMultibandTopomaps
from eeg_affect.models.generator import EEGFeatureGenerator

print("EEG Affect modules imported successfully with camelCase API!")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Loading and Subject Structure
We load the 462-dimensional feature matrix extracted from 14-channel Emotiv EEG across 88 participants and 27 Cowen emotion classes.
"""))

nb.cells.append(nbf.v4.new_code_cell("""X_df, yCowen, groups, metadata = loadEegFeatures(target='cowen')
print(f"Total Samples: {len(X_df)}")
print(f"Feature Dimensions: {X_df.shape[1]} (33 features x 14 channels)")
print(f"Participants: {len(np.unique(groups))}")
print(f"Emotions: {len(np.unique(yCowen))}")
metadata.head()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Frontal Alpha Asymmetry (FAA)
Frontal Alpha Asymmetry (AF4 vs AF3) is a foundational biomarker for approach (positive valence) vs. avoidance (negative valence) motivation:
$$\\mathrm{FAA} = \\ln(\\mathrm{Alpha}_{\\mathrm{AF4}}) - \\ln(\\mathrm{Alpha}_{\\mathrm{AF3}})$$
"""))

nb.cells.append(nbf.v4.new_code_cell("""faa = computeFrontalAlphaAsymmetry(X_df, 'AF3', 'AF4')
faaDf = pd.DataFrame({'CowenID': metadata['CowenID'], 'FAA': faa})
faaDf['Emotion'] = faaDf['CowenID'].map(COWEN_27_EMOTIONS)

faaSummary = faaDf.groupby('Emotion')['FAA'].agg(['mean', 'sem']).sort_values('mean')

plt.figure(figsize=(10, 6), dpi=150)
colors = ['#e74c3c' if m < 0 else '#2ecc71' for m in faaSummary['mean']]
plt.barh(faaSummary.index, faaSummary['mean'], xerr=faaSummary['sem'], color=colors, alpha=0.85, capsize=3)
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title('Frontal Alpha Asymmetry (AF4 - AF3) Across 27 Emotional States', fontweight='bold')
plt.xlabel('FAA Metric (Approach vs Withdrawal)', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Manifold Learning and Representational Similarity Analysis (RSA)
We compute emotion centroids, evaluate whether the neural geometry correlates with psychological Valence-Arousal space, and visualize the manifold using Diffusion / Laplacian Eigenmaps.
"""))

nb.cells.append(nbf.v4.new_code_cell("""preprocessor = EEGPreprocessor(scaler='robust')
X_scaled = preprocessor.fitTransform(X_df)

cowenIds1Based = yCowen + 1
centroids, uniqueEids, emotionNames = computeEmotionCentroids(X_scaled, cowenIds1Based)
distMatrix = computePairwiseGeometry(centroids, metric='euclidean')

rsaMetrics = compareEegVsPsychologyGeometry(centroids, uniqueEids)
print(f"RSA Pearson r:    {rsaMetrics['pearson_r']:.4f} (p = {rsaMetrics['pearson_p']:.4e})")
print(f"RSA Spearman rho: {rsaMetrics['spearman_rho']:.4f} (p = {rsaMetrics['spearman_p']:.4e})")

embedder = EmotionManifoldEmbedder(method='diffusion', n_components=2, n_neighbors=15)
coordsDiff = embedder.fitTransform(centroids)

plt.figure(figsize=(9, 7), dpi=150)
plt.scatter(coordsDiff[:, 0], coordsDiff[:, 1], c='purple', s=80, alpha=0.8)
for i, name in enumerate(emotionNames):
    plt.annotate(name, (coordsDiff[i, 0] + 0.002, coordsDiff[i, 1] + 0.002), fontsize=8)
plt.title('Diffusion Map of 27 Emotion Centroids in Neural State Space', fontweight='bold')
plt.xlabel('Diffusion Coordinate 1')
plt.ylabel('Diffusion Coordinate 2')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Class-Conditional Synthetic EEG Feature Generation
We fit multivariate Gaussian models with Ledoit-Wolf shrinkage covariance estimation to sample class-conditional synthetic EEG features.
"""))

nb.cells.append(nbf.v4.new_code_cell("""gen = EEGFeatureGenerator(random_state=42)
gen.fit(X_scaled[:5000], yCowen[:5000])

# Sample 50 synthetic Joy samples (index 19)
joySynth = gen.sample(19, n_samples=50)
print(f"Generated synthetic Joy samples shape: {joySynth.shape}")

# Compute Frechet Distance against empirical test samples
fdJoy = gen.computeFrechetDistance(19, X_scaled[5000:][yCowen[5000:] == 19])
print(f"Frechet Distance for Joy: {fdJoy:.3f}")
"""))

with open(nbPath, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated notebook with camelCase: {nbPath}")
