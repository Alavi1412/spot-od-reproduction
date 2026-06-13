#!/usr/bin/env python
"""Train and evaluate hybrid ablations."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import (
    build_innovation_features,
    parse_baseline_config,
    relative_improvement_percent,
    run_filter_baselines,
    run_model_inference,
    score_predictions,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.seeding import seed_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--output-dir", type=str, default="results/ablation")
    p.add_argument("--baseline-cache-dir", type=str, default="results/baseline_cache")
    p.add_argument(
        "--variants",
        type=str,
        default="HybridGNN_Full,Hybrid_InnovationConditioned,Hybrid_NoGraph,Hybrid_NoGate,Hybrid_NoBound,Hybrid_NoReg",
    )
    p.add_argument("--skip-existing", action="store_true")
    return p


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_or_compute_baselines(cfg: dict, baseline_cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_dir = Path(cfg["data"]["output_dir"])
    test_arr = load_dataset_npz(data_dir / "test.npz")
    stress_arr = load_dataset_npz(data_dir / "stress_test.npz")
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    test_cache = baseline_cache_dir / "test_baselines.npz"
    stress_cache = baseline_cache_dir / "stress_test_baselines.npz"
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)

    if test_cache.exists():
        t = np.load(test_cache)
        test_pred = {"ekf": t["ekf"], "ukf": t["ukf"]}
        if "aukf" in t.files:
            test_pred["aukf"] = t["aukf"]
        needs_aukf = baseline_cfg.aukf is not None and "aukf" not in test_pred
        if test_pred["ekf"].shape != test_arr.states.shape or test_pred["ukf"].shape != test_arr.states.shape or needs_aukf:
            test_pred = run_filter_baselines(
                states=test_arr.states,
                measurements=test_arr.measurements,
                visibility=test_arr.visibility,
                times=test_arr.times,
                dataset_cfg=parse_dataset_config(cfg["simulation"]),
                baseline_cfg=baseline_cfg,
                seed=int(cfg["seed"]) + 3,
                x0_estimates=test_arr.x0_estimates,
            )
            np.savez_compressed(test_cache, **test_pred)
    else:
        test_pred = run_filter_baselines(
            states=test_arr.states,
            measurements=test_arr.measurements,
            visibility=test_arr.visibility,
            times=test_arr.times,
            dataset_cfg=parse_dataset_config(cfg["simulation"]),
            baseline_cfg=baseline_cfg,
            seed=int(cfg["seed"]) + 3,
            x0_estimates=test_arr.x0_estimates,
        )
        np.savez_compressed(test_cache, **test_pred)

    stress_cfg = deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])
    if stress_cache.exists():
        s = np.load(stress_cache)
        stress_pred = {"ekf": s["ekf"], "ukf": s["ukf"]}
        if "aukf" in s.files:
            stress_pred["aukf"] = s["aukf"]
        needs_aukf = baseline_cfg.aukf is not None and "aukf" not in stress_pred
        if stress_pred["ekf"].shape != stress_arr.states.shape or stress_pred["ukf"].shape != stress_arr.states.shape or needs_aukf:
            stress_pred = run_filter_baselines(
                states=stress_arr.states,
                measurements=stress_arr.measurements,
                visibility=stress_arr.visibility,
                times=stress_arr.times,
                dataset_cfg=parse_dataset_config(stress_cfg),
                baseline_cfg=baseline_cfg,
                seed=int(cfg["seed"]) + 17,
                x0_estimates=stress_arr.x0_estimates,
            )
            np.savez_compressed(stress_cache, **stress_pred)
    else:
        stress_pred = run_filter_baselines(
            states=stress_arr.states,
            measurements=stress_arr.measurements,
            visibility=stress_arr.visibility,
            times=stress_arr.times,
            dataset_cfg=parse_dataset_config(stress_cfg),
            baseline_cfg=baseline_cfg,
            seed=int(cfg["seed"]) + 17,
            x0_estimates=stress_arr.x0_estimates,
        )
        np.savez_compressed(stress_cache, **stress_pred)

    return test_pred, stress_pred


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    base_seed = int(cfg["seed"])

    data_dir = Path(cfg["data"]["output_dir"])
    train_arr = load_dataset_npz(data_dir / "train.npz")
    val_arr = load_dataset_npz(data_dir / "val.npz")
    test_arr = load_dataset_npz(data_dir / "test.npz")
    stress_arr = load_dataset_npz(data_dir / "stress_test.npz")

    test_baseline, stress_baseline = load_or_compute_baselines(cfg, Path(args.baseline_cache_dir))
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    test_ukf = score_predictions(test_arr.states[:, eval_start:], test_baseline["ukf"][:, eval_start:])
    stress_ukf = score_predictions(stress_arr.states[:, eval_start:], stress_baseline["ukf"][:, eval_start:])
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    stress_dataset_cfg = parse_dataset_config(deep_update(cfg["simulation"], cfg["stress_simulation_overrides"]))
    test_innovation_features = test_arr.innovation_features
    if test_innovation_features is None:
        test_innovation_features = build_innovation_features(
            dataset_cfg=dataset_cfg,
            measurements=test_arr.measurements,
            visibility=test_arr.visibility,
            times=test_arr.times,
            ekf_prior=test_baseline["ekf"],
        )
    stress_innovation_features = stress_arr.innovation_features
    if stress_innovation_features is None:
        stress_innovation_features = build_innovation_features(
            dataset_cfg=stress_dataset_cfg,
            measurements=stress_arr.measurements,
            visibility=stress_arr.visibility,
            times=stress_arr.times,
            ekf_prior=stress_baseline["ekf"],
        )

    available_variants: dict[str, dict] = {
        "HybridGNN_Full": {
            "name": "HybridGNN_Full",
            "model_kwargs": {"residual_scale": 0.02, "use_gating": True, "bounded_residual": True},
            "residual_reg_weight": 0.05,
            "gnn_layers": train_cfg.gnn_layers,
        },
        "Hybrid_NoGraph": {
            "name": "Hybrid_NoGraph",
            "model_kwargs": {"residual_scale": 0.02, "use_gating": True, "bounded_residual": True},
            "residual_reg_weight": 0.05,
            "gnn_layers": 0,
        },
        "Hybrid_InnovationConditioned": {
            "name": "Hybrid_InnovationConditioned",
            "model_kwargs": {
                "residual_scale": 0.03,
                "use_gating": True,
                "bounded_residual": True,
                "use_innovation_features": True,
                "use_context_budget": True,
                "use_dual_prior_fusion": True,
            },
            "residual_reg_weight": 0.05,
            "gnn_layers": train_cfg.gnn_layers,
        },
        "Hybrid_NoGate": {
            "name": "Hybrid_NoGate",
            "model_kwargs": {"residual_scale": 0.02, "use_gating": False, "bounded_residual": True},
            "residual_reg_weight": 0.05,
            "gnn_layers": train_cfg.gnn_layers,
        },
        "Hybrid_NoBound": {
            "name": "Hybrid_NoBound",
            "model_kwargs": {"residual_scale": 0.002, "use_gating": True, "bounded_residual": False},
            "residual_reg_weight": 0.05,
            "gnn_layers": train_cfg.gnn_layers,
        },
        "Hybrid_NoReg": {
            "name": "Hybrid_NoReg",
            "model_kwargs": {"residual_scale": 0.02, "use_gating": True, "bounded_residual": True},
            "residual_reg_weight": 0.0,
            "gnn_layers": train_cfg.gnn_layers,
        },
    }
    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    variants: list[dict] = []
    for name in variant_names:
        if name not in available_variants:
            raise ValueError(f"Unknown variant '{name}'. Available: {sorted(available_variants)}")
        variants.append(available_variants[name])

    prior_rows: dict[str, dict] = {}
    csv_path = out_dir / "ablation_metrics.csv"
    if args.skip_existing and csv_path.exists():
        old = pd.read_csv(csv_path)
        for _, r in old.iterrows():
            prior_rows[str(r["variant"])] = r.to_dict()

    rows: list[dict] = []
    for name in variant_names:
        if name in prior_rows and args.skip_existing:
            rows.append(prior_rows[name])

    variants_to_run: list[dict] = []
    for var in variants:
        vname = str(var["name"])
        if args.skip_existing and vname in prior_rows and (out_dir / vname / "best_hybrid.pt").exists():
            print(f"Skipping existing variant: {vname}")
            continue
        variants_to_run.append(var)

    for i, var in enumerate(variants_to_run):
        seed = base_seed + i
        print(f"\n=== {var['name']} (seed={seed}) ===")
        seed_all(seed)
        var_dir = out_dir / var["name"]
        var_train_cfg = replace(train_cfg, gnn_layers=int(var.get("gnn_layers", train_cfg.gnn_layers)))
        model, hist, ckpt = train_model(
            train_arrays=train_arr,
            val_arrays=val_arr,
            cfg=var_train_cfg,
            output_dir=var_dir,
            seed=seed,
            use_ekf_prior=True,
            model_kwargs=var["model_kwargs"],
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
        )

        pred_test = run_model_inference(
            model=model,
            states=test_arr.states,
            measurements=test_arr.measurements,
            visibility=test_arr.visibility,
            station_ecef=test_arr.station_ecef,
            window_size=var_train_cfg.window_size,
            ekf_prior=test_baseline["ekf"] if (model.use_ekf_prior or getattr(model, "use_prior_bank_fusion", False)) else None,
            ukf_prior=test_baseline.get("ukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            aukf_prior=test_baseline.get("aukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            secondary_prior=test_baseline.get("aukf") if model.use_dual_prior_fusion else None,
            innovation_features=test_innovation_features if model.use_innovation_features else None,
            prior_bank_stats=test_arr.prior_bank_stats if getattr(model, "use_prior_bank_fusion", False) else None,
        )
        pred_stress = run_model_inference(
            model=model,
            states=stress_arr.states,
            measurements=stress_arr.measurements,
            visibility=stress_arr.visibility,
            station_ecef=stress_arr.station_ecef,
            window_size=var_train_cfg.window_size,
            ekf_prior=stress_baseline["ekf"] if (model.use_ekf_prior or getattr(model, "use_prior_bank_fusion", False)) else None,
            ukf_prior=stress_baseline.get("ukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            aukf_prior=stress_baseline.get("aukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            secondary_prior=stress_baseline.get("aukf") if model.use_dual_prior_fusion else None,
            innovation_features=stress_innovation_features if model.use_innovation_features else None,
            prior_bank_stats=stress_arr.prior_bank_stats if getattr(model, "use_prior_bank_fusion", False) else None,
        )

        m_test = score_predictions(test_arr.states[:, eval_start:], pred_test[:, eval_start:])
        m_stress = score_predictions(stress_arr.states[:, eval_start:], pred_stress[:, eval_start:])
        rows.append(
            {
                "variant": var["name"],
                "seed": seed,
                "checkpoint": str(ckpt),
                "epochs_trained": len(hist["train_loss"]),
                "test_pos_rmse_m": m_test["pos_rmse_m"],
                "test_vel_rmse_mps": m_test["vel_rmse_mps"],
                "stress_pos_rmse_m": m_stress["pos_rmse_m"],
                "stress_vel_rmse_mps": m_stress["vel_rmse_mps"],
                "test_improvement_vs_ukf_percent": relative_improvement_percent(
                    test_ukf["pos_rmse_m"], m_test["pos_rmse_m"]
                ),
                "stress_improvement_vs_ukf_percent": relative_improvement_percent(
                    stress_ukf["pos_rmse_m"], m_stress["pos_rmse_m"]
                ),
                "residual_reg_weight": var["residual_reg_weight"],
                "residual_scale": var["model_kwargs"]["residual_scale"],
                "use_gating": var["model_kwargs"]["use_gating"],
                "bounded_residual": var["model_kwargs"]["bounded_residual"],
                "gnn_layers": int(var.get("gnn_layers", train_cfg.gnn_layers)),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["variant"], keep="last")
    df.to_csv(out_dir / "ablation_metrics.csv", index=False)
    best_idx = int(df["stress_pos_rmse_m"].idxmin())
    summary = {
        "best_variant_by_stress_pos_rmse": df.iloc[best_idx]["variant"],
        "rows": rows,
    }
    dump_json(summary, out_dir / "ablation_summary.json")
    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
