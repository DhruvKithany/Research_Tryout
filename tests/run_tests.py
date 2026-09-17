"""Test runner using Python's built-in unittest."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_data_loader import (
    test_load_channel_locations,
    test_load_eeg_features_cowen,
    test_load_eeg_features_quadrant,
    test_load_participants_info,
    test_subject_independent_split,
)
from tests.test_features import (
    test_feature_groupings,
    test_frontal_alpha_asymmetry,
    test_signal_statistics,
)
from tests.test_geometry import (
    test_centroids_and_pairwise_distance,
    test_manifold_embedder,
    test_two_nn_intrinsic_dimension,
)
from tests.test_models import (
    test_baseline_models,
    test_cosine_nearest_centroid_proba,
    test_eeg_feature_generator,
    test_neural_model,
)


def run_all_tests():
    print("=" * 70)
    print("Running EEG Affect Test Suite")
    print("=" * 70)

    test_funcs = [
        ("test_load_channel_locations", test_load_channel_locations),
        ("test_load_participants_info", test_load_participants_info),
        ("test_load_eeg_features_cowen", test_load_eeg_features_cowen),
        ("test_load_eeg_features_quadrant", test_load_eeg_features_quadrant),
        ("test_subject_independent_split", test_subject_independent_split),
        ("test_frontal_alpha_asymmetry", test_frontal_alpha_asymmetry),
        ("test_feature_groupings", test_feature_groupings),
        ("test_signal_statistics", test_signal_statistics),
        ("test_manifold_embedder", test_manifold_embedder),
        ("test_centroids_and_pairwise_distance", test_centroids_and_pairwise_distance),
        ("test_two_nn_intrinsic_dimension", test_two_nn_intrinsic_dimension),
        ("test_baseline_models", test_baseline_models),
        ("test_cosine_nearest_centroid_proba", test_cosine_nearest_centroid_proba),
        ("test_neural_model", test_neural_model),
        ("test_eeg_feature_generator", test_eeg_feature_generator),
    ]

    passed = 0
    failed = 0
    for name, fn in test_funcs:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"Tests Completed: {passed} passed, {failed} failed.")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
