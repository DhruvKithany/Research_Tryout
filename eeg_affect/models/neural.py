"""Multi-Layer Perceptron (MLP) architecture with regularization and early stopping."""

from typing import Tuple

from sklearn.neural_network import MLPClassifier


def buildNeuralModel(
    hiddenLayerSizes: Tuple[int, ...] = (256, 128, 64),
    alpha: float = 1e-4,
    learningRateInit: float = 1e-3,
    maxIter: int = 150,
    randomState: int = 42,
    **kwargs,
) -> MLPClassifier:
    """Creates a funnel-shaped Multi-Layer Perceptron classifier."""
    hSizes = kwargs.get("hidden_layer_sizes", hiddenLayerSizes)
    lrInit = kwargs.get("learning_rate_init", learningRateInit)
    mIter = kwargs.get("max_iter", maxIter)
    rState = kwargs.get("random_state", randomState)

    return MLPClassifier(
        hidden_layer_sizes=hSizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=lrInit,
        max_iter=mIter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=rState,
    )


# Aliases for backward compatibility
build_neural_model = buildNeuralModel
