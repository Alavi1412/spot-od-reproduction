"""Faithful KalmanNet reproduction harness (Loop 41).

Drives the official KalmanNet_TSP source (vendored under
external/third_party/KalmanNet_TSP, upstream commit pinned in
external/third_party/KalmanNet_TSP_COMMIT) on a small linear-canonical
sanity-check benchmark.

The point is *not* to retrain the full canonical configuration from the
official paper; it is to (a) drive the official architecture and pipeline
end-to-end exactly as released, and (b) record a sanity check against the
expected qualitative ordering documented in the official source release:

    observation noise floor  >  KalmanNet test MSE  >=  optimal Kalman MSE

If KalmanNet's trained MSE comes out at or below the observation noise
floor and within a small dB margin of the optimal Kalman filter on the
same generated data, the reproduction has reproduced the qualitative
behaviour of the official benchmark with the official code.

Outputs JSON to results/kalmannet_repro/sanity_check.json so a downstream
table generator can ingest the result without re-importing torch.

This file is non-paper-facing: it is a reproduction harness. It does not
make any paper claim.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = REPO_ROOT / "external" / "third_party" / "KalmanNet_TSP"
if not VENDOR_ROOT.exists():
    raise SystemExit(
        f"Vendored KalmanNet_TSP not found at {VENDOR_ROOT}. "
        "Run the vendor clone first."
    )

# Inject the vendored source root onto sys.path so imports work
# exactly as the official main scripts.
sys.path.insert(0, str(VENDOR_ROOT))
os.chdir(str(VENDOR_ROOT))  # the official scripts use relative paths

import torch  # noqa: E402

# Compatibility: KalmanNet_TSP saves and reloads pickled module objects via
# torch.save/torch.load. PyTorch >= 2.6 made torch.load default to
# weights_only=True, which rejects arbitrary pickled classes. We are loading
# only a checkpoint we just wrote in-process, so it is safe to restore the
# pre-2.6 behaviour locally for the vendored pipeline.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # type: ignore[assignment]

from Simulations.Linear_sysmdl import SystemModel  # noqa: E402
from Simulations.utils import DataGen  # noqa: E402
from Simulations.Linear_canonical.parameters import (  # noqa: E402
    F,
    H,
    Q_structure,
    R_structure,
    m,
    m1_0,
)
from Filters.KalmanFilter_test import KFTest  # noqa: E402
from KNet.KalmanNet_nn import KalmanNetNN  # noqa: E402
from Pipelines.Pipeline_EKF import Pipeline_EKF  # noqa: E402


def _make_args(n_steps: int, n_batch: int, lr: float, wd: float,
               N_E: int, N_CV: int, N_T: int, T: int):
    class _Args:
        pass

    a = _Args()
    a.N_E = N_E
    a.N_CV = N_CV
    a.N_T = N_T
    a.T = T
    a.T_test = T
    a.randomLength = False
    a.T_max = T
    a.T_min = T
    a.randomInit_train = False
    a.randomInit_cv = False
    a.randomInit_test = False
    a.variance = 1.0
    a.distribution = "normal"
    a.use_cuda = False
    a.n_steps = n_steps
    a.n_batch = n_batch
    a.lr = lr
    a.wd = wd
    a.CompositionLoss = False
    a.alpha = 0.3
    a.in_mult_KNet = 5
    a.out_mult_KNet = 40
    return a


def main(out_json: Path,
         n_steps: int = 300,
         n_batch: int = 20,
         lr: float = 1e-3,
         wd: float = 1e-3,
         N_E: int = 200,
         N_CV: int = 50,
         N_T: int = 50,
         T: int = 30,
         seed: int = 41) -> int:
    torch.manual_seed(seed)
    args = _make_args(n_steps=n_steps, n_batch=n_batch, lr=lr, wd=wd,
                      N_E=N_E, N_CV=N_CV, N_T=N_T, T=T)
    device = torch.device("cpu")

    # Replicate main_linear_canonical's noise scaling.
    r2 = torch.tensor([1.0])
    vdB = -20.0
    v = 10.0 ** (vdB / 10.0)
    q2 = v * r2

    Q = q2 * Q_structure
    R = r2 * R_structure
    sys_model = SystemModel(F, Q, H, R, args.T, args.T_test)
    m2_0 = torch.zeros(m, m)
    sys_model.InitSequence(m1_0, m2_0)

    data_dir = Path("Simulations/Linear_canonical/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "loop41_smoke.pt"
    DataGen(args, sys_model, str(data_file))
    bundle = torch.load(str(data_file), map_location=device)
    train_input, train_target, cv_input, cv_target, test_input, test_target = bundle[:6]

    import torch.nn as _nn  # local import to keep top imports clean
    loss_obs = _nn.MSELoss(reduction="mean")
    obs_arr = torch.empty(args.N_T)
    for i in range(args.N_T):
        obs_arr[i] = loss_obs(test_input[i], test_target[i]).item()
    obs_mse = obs_arr.mean().item()
    obs_db = 10.0 * torch.log10(torch.tensor(obs_mse)).item()

    _kf_arr, _kf_avg, kf_db, _kf_out = KFTest(args, sys_model, test_input, test_target)
    kf_db = float(kf_db.item() if hasattr(kf_db, "item") else kf_db)

    knet = KalmanNetNN()
    knet.NNBuild(sys_model, args)
    n_params = sum(p.numel() for p in knet.parameters() if p.requires_grad)

    pipe = Pipeline_EKF("loop41_smoke", "KNet", "KalmanNet")
    pipe.setssModel(sys_model)
    pipe.setModel(knet)
    pipe.setTrainingParams(args)

    knet_dir = Path("KNet")
    knet_dir.mkdir(exist_ok=True)
    path_results = "KNet/"

    t0 = time.time()
    pipe.NNTrain(sys_model, cv_input, cv_target, train_input, train_target,
                 path_results=path_results)
    train_secs = time.time() - t0

    _knet_arr, _knet_avg, knet_db, _knet_out, _knet_t = pipe.NNTest(
        sys_model, test_input, test_target, path_results=path_results)
    knet_db = float(knet_db.item() if hasattr(knet_db, "item") else knet_db)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vendor_path": str(VENDOR_ROOT.relative_to(REPO_ROOT)),
        "config": {
            "state_dim": int(m),
            "T": int(args.T),
            "N_E": int(args.N_E),
            "N_CV": int(args.N_CV),
            "N_T": int(args.N_T),
            "n_steps": int(args.n_steps),
            "n_batch": int(args.n_batch),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "vdB": float(vdB),
            "seed": int(seed),
            "device": "cpu",
        },
        "n_params_kalmannet": int(n_params),
        "observation_floor_db": float(obs_db),
        "optimal_kf_db": float(kf_db),
        "kalmannet_db": float(knet_db),
        "knet_train_seconds": float(train_secs),
        "sanity_checks": {
            "knet_below_obs_floor": bool(knet_db < obs_db),
            "knet_within_2db_of_kf": bool(abs(knet_db - kf_db) <= 2.0),
            "knet_within_5db_of_kf": bool(abs(knet_db - kf_db) <= 5.0),
        },
        "expected_qualitative_ordering": (
            "observation_floor > kalmannet_test_mse >= optimal_kf_mse"
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    target = REPO_ROOT / "results" / "kalmannet_repro" / "sanity_check.json"
    raise SystemExit(main(target))
