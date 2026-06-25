from __future__ import annotations

from pathlib import Path

from scripts.build_trajectory_residual_refine_figure import GRAPH_DIR, LOCAL_DIR, MEAN_DIR, OUTPUT_PATH


def test_default_release_paths_target_validation_selected_val53_artifacts() -> None:
    assert GRAPH_DIR == Path(
        "results/"
        "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
        "newfresh151157163167_20260625"
    )
    assert LOCAL_DIR == Path(
        "results/"
        "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
        "newfresh151157163167_20260625"
    )
    assert MEAN_DIR == Path(
        "results/"
        "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
        "newfresh151157163167_20260625"
    )
    assert OUTPUT_PATH == Path("paper/figures/trajectory_residual_refine_gain_distribution_val53.png")
