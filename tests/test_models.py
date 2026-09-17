"""Unit tests for classifiers, ensembles, neural model, and generator."""

import numpy as np

from eeg_affect.models.baselines import CosineNearestCentroid, build_baseline_models
from eeg_affect.models.ensembles import build_ensemble_models
from eeg_affect.models.generator import EEGFeatureGenerator
from eeg_affect.models.neural import build_neural_model


def test_baseline_models():
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 10)
    y_train = rng.choice([0, 1, 2, 3], size=100)
    X_test = rng.randn(20, 10)

    models = build_baseline_models(random_state=42)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == 20


def test_cosine_nearest_centroid_proba():
    rng = np.random.RandomState(42)
    X_train = rng.randn(60, 8)
    y_train = rng.choice([0, 1, 2], size=60)
    X_test = rng.randn(10, 8)

    cnc = CosineNearestCentroid()
    cnc.fit(X_train, y_train)
    proba = cnc.predict_proba(X_test)
    assert proba.shape == (10, 3)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_neural_model():
    rng = np.random.RandomState(42)
    X_train = rng.randn(80, 12)
    y_train = rng.choice([0, 1], size=80)
    X_test = rng.randn(15, 12)

    mlp = build_neural_model(hidden_layer_sizes=(32, 16), max_iter=20, random_state=42)
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)
    assert len(preds) == 15


def test_eeg_feature_generator():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = rng.choice([0, 1], size=100)

    gen = EEGFeatureGenerator(random_state=42)
    gen.fit(X, y)

    # Sample class 0
    samples_0 = gen.sample(0, n_samples=15)
    assert samples_0.shape == (15, 10)

    # Sample all
    X_synth, y_synth = gen.sample_all(n_samples_per_class=10)
    assert X_synth.shape == (20, 10)
    assert len(y_synth) == 20

    # Frechet distance
    fd = gen.compute_frechet_distance(0, X[y == 0])
    assert fd >= 0.0
