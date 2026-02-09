#!/usr/bin/env python3
"""
reproduce.py

Reproducibility script for:
"Reproducible QUBO--QAOA Benchmark for Cardinality-Constrained Mean--Variance Portfolio Selection"

Outputs (written to current working directory):
- results_instances.csv
- results_summary.csv
- results_ablation_topM.csv
- results_alpha_sensitivity.csv
- fig_gap_boxplot.pdf
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_instance(N: int, seed: int, var_max: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.15, size=N)
    A = rng.normal(0.0, 1.0, size=(N, N))
    Sigma = A @ A.T
    diag_max = float(np.max(np.diag(Sigma)))
    Sigma = Sigma * (var_max / diag_max)
    return mu, Sigma


def penalty_A(mu: np.ndarray, Sigma: np.ndarray, lam: float, K: int, alpha: float) -> float:
    mu_max = float(np.max(mu))
    eigmax = float(np.linalg.eigvalsh(Sigma).max())
    return float(alpha * (mu_max + lam * (eigmax / K)))


def energy(mu: np.ndarray, Sigma: np.ndarray, lam: float, K: int, Apen: float, x: np.ndarray) -> float:
    x = x.astype(float, copy=False)
    J = (mu @ x) / K - lam * (x @ Sigma @ x) / (K ** 2)
    pen = Apen * (float(x.sum()) - K) ** 2
    return float(-J + pen)


def feasible_opt(mu: np.ndarray, Sigma: np.ndarray, lam: float, K: int) -> Tuple[float, np.ndarray]:
    N = len(mu)
    bestE = None
    bestx = None
    for comb in itertools.combinations(range(N), K):
        x = np.zeros(N, dtype=int)
        x[list(comb)] = 1
        E = energy(mu, Sigma, lam, K, 0.0, x)
        if bestE is None or E < bestE:
            bestE = E
            bestx = x.copy()
    return float(bestE), bestx


def energies_all(mu: np.ndarray, Sigma: np.ndarray, lam: float, K: int, Apen: float) -> np.ndarray:
    N = len(mu)
    E = np.empty(2 ** N, dtype=float)
    for z in range(2 ** N):
        x = np.fromiter(((z >> i) & 1 for i in range(N)), dtype=int, count=N)
        E[z] = energy(mu, Sigma, lam, K, Apen, x)
    return E


def apply_mixer(state: np.ndarray, beta: float, N: int) -> np.ndarray:
    c = math.cos(beta)
    s = math.sin(beta)
    for q in range(N):
        state = state.reshape((2 ** q, 2, 2 ** (N - q - 1)))
        a0 = state[:, 0, :].copy()
        a1 = state[:, 1, :].copy()
        state[:, 0, :] = c * a0 + (-1j * s) * a1
        state[:, 1, :] = (-1j * s) * a0 + c * a1
        state = state.reshape((2 ** N,))
    return state


def qaoa_p1(Etable: np.ndarray, gammas: np.ndarray, betas: np.ndarray, topM: int = 256):
    N = int(round(math.log2(len(Etable))))
    dim = 2 ** N
    state0 = np.ones(dim, dtype=complex) / math.sqrt(dim)

    best_expE = None
    best_params = None
    best_probs = None

    for gamma in gammas:
        stateC = state0 * np.exp(-1j * gamma * Etable)
        for beta in betas:
            st = apply_mixer(stateC.copy(), beta, N)
            probs = (st.conj() * st).real
            expE = float(np.dot(probs, Etable))
            if best_expE is None or expE < best_expE:
                best_expE = expE
                best_params = (float(gamma), float(beta))
                best_probs = probs

    order = np.argsort(best_probs)[::-1]
    M = min(int(topM), len(order))
    top = order[:M]
    bestz = int(top[np.argmin(Etable[top])])
    return bestz, best_params, float(best_expE), best_probs


def z_to_x(z: int, N: int) -> np.ndarray:
    return np.fromiter(((z >> i) & 1 for i in range(N)), dtype=int, count=N)


def run_benchmark(
    N_list=(8, 10, 12),
    K: int = 3,
    lam: float = 0.6,
    alpha: float = 10.0,
    n_instances: int = 10,
    grid_n: int = 21,
    topM: int = 256,
    seed0: int = 1234,
):
    gammas = np.linspace(0.0, 2.0 * math.pi, grid_n)
    betas = np.linspace(0.0, math.pi, grid_n)

    rows = []
    gaps_by_N: Dict[int, List[float]] = {}

    for N in N_list:
        gaps: List[float] = []
        for i in range(n_instances):
            seed = int(seed0 + 1000 * N + i)
            mu, Sigma = generate_instance(N, seed)
            Apen = penalty_A(mu, Sigma, lam, K, alpha)
            Etable = energies_all(mu, Sigma, lam, K, Apen)
            optE, _ = feasible_opt(mu, Sigma, lam, K)

            zhat, params, _, _ = qaoa_p1(Etable, gammas, betas, topM=topM)
            xhat = z_to_x(zhat, N)

            # Report energy on the feasible objective (-J), i.e., with penalty removed.
            Esel = energy(mu, Sigma, lam, K, 0.0, xhat)

            gap = (Esel - optE) / abs(optE) if optE != 0 else 0.0
            gaps.append(float(gap))

            rows.append(
                dict(
                    N=N,
                    instance=i,
                    seed=seed,
                    gamma=params[0],
                    beta=params[1],
                    selected_card=int(xhat.sum()),
                    feasible=int(xhat.sum() == K),
                    E_opt=float(optE),
                    E_sel=float(Esel),
                    gap=float(gap),
                )
            )
        gaps_by_N[N] = gaps

    df = pd.DataFrame(rows)
    summary_rows = []
    for N in N_list:
        sub = df[df["N"] == N].copy()
        g = np.asarray(gaps_by_N[N], dtype=float)
        summary_rows.append(
            dict(
                N=N,
                mean_gap=float(g.mean()),
                median_gap=float(np.median(g)),
                std_gap=float(g.std(ddof=0)),
                exact_hit=float((np.abs(sub["E_sel"] - sub["E_opt"]) < 1e-9).mean()),
                feasible_rate=float(sub["feasible"].mean()),
                topM=int(topM),
                alpha=float(alpha),
                grid_n=int(grid_n),
                n_instances=int(n_instances),
            )
        )
    summary = pd.DataFrame(summary_rows)
    return df, summary, gaps_by_N


def make_boxplot(gaps_by_N: Dict[int, List[float]], outpath: str = "fig_gap_boxplot.pdf"):
    Ns = sorted(gaps_by_N.keys())
    data = [gaps_by_N[N] for N in Ns]
    plt.figure()
    plt.boxplot(data, labels=[str(N) for N in Ns])
    plt.xlabel("Number of assets N")
    plt.ylabel("Relative optimality gap")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def run_ablation_topM(M_list=(1, 16, 256), **kwargs) -> pd.DataFrame:
    summaries = []
    for M in M_list:
        _, summary, _ = run_benchmark(topM=M, **kwargs)
        summaries.append(summary[["N", "mean_gap", "median_gap", "std_gap", "exact_hit", "feasible_rate"]].assign(topM=int(M)))
    return pd.concat(summaries, ignore_index=True).sort_values(["topM", "N"])


def run_alpha_sensitivity(alpha_list=(1, 2, 5, 10, 20), N: int = 12, topM: int = 256, **kwargs) -> pd.DataFrame:
    rows = []
    for a in alpha_list:
        _, summary, _ = run_benchmark(N_list=(N,), alpha=float(a), topM=int(topM), **kwargs)
        r = summary.iloc[0][["mean_gap", "median_gap", "std_gap", "exact_hit", "feasible_rate"]].to_dict()
        r["alpha"] = float(a)
        r["N"] = int(N)
        r["topM"] = int(topM)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["alpha"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default=".", help="Output directory")
    p.add_argument("--seed0", type=int, default=1234, help="Base seed for the fixed benchmark schedule")
    args = p.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df, summary, gaps = run_benchmark(seed0=args.seed0)
    df.to_csv(os.path.join(outdir, "results_instances.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "results_summary.csv"), index=False)

    make_boxplot(gaps, outpath=os.path.join(outdir, "fig_gap_boxplot.pdf"))

    abM = run_ablation_topM(seed0=args.seed0)
    abM.to_csv(os.path.join(outdir, "results_ablation_topM.csv"), index=False)

    abA = run_alpha_sensitivity(seed0=args.seed0)
    abA.to_csv(os.path.join(outdir, "results_alpha_sensitivity.csv"), index=False)

    print("Wrote outputs to:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
