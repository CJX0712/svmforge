from core.config import Config
from data.synthetic import available_dgps
from pipeline.train_eval import (
    FLAGSHIP,
    SINGLE_SOTA,
    run_benchmark,
    verify_determinism,
)


def test_run_benchmark_small():
    cfg = Config(seed=1, n_train=200, n_test=100, n_val=50, n_seeds=2)
    res = run_benchmark(["moons", "circles"], [0, 1], cfg)
    assert "summary" in res
    assert "gate" in res["summary"]
    assert set(res["summary"]["per_method"].keys()) >= {
        "lr",
        "knn",
        "svc_rbf",
        "lsvm_rbf",
        "krr_rbf",
        FLAGSHIP,
    }
    # every method has a finite mean accuracy recorded
    for m, d in res["summary"]["per_method"].items():
        assert d["n"] == 4  # 2 datasets x 2 seeds


def test_benchmark_is_deterministic():
    cfg = Config(seed=2, n_train=200, n_test=100, n_val=50, n_seeds=2)
    r1 = run_benchmark(["moons"], [0, 1], cfg)
    r2 = run_benchmark(["moons"], [0, 1], cfg)
    assert verify_determinism(r1, r2) is True


def test_gate_keys_present():
    cfg = Config(seed=3, n_train=200, n_test=100, n_val=50, n_seeds=2)
    res = run_benchmark(available_dgps()[:3], [0, 1], cfg)
    g = res["summary"]["gate"]
    for k in [
        "flagship_mean_acc",
        "weak_naive_lr_mean_acc",
        "margin_vs_lr",
        "gate1_beat_lr_pass",
        "strong_naive_knn_mean_acc",
        "noninferior_vs_knn",
        "knn_win_tie_rate",
        "knn_noninferior",
        "single_sota_svc_mean_acc",
        "noninferior_vs_svc",
        "gate2_pass",
        "all_pass",
    ]:
        assert k in g
    assert SINGLE_SOTA in res["summary"]["per_method"]
