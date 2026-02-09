# Reproducible QUBO–QAOA Portfolio Benchmark (ICECET 2026)

This repository contains a fully reproducible simulator-based benchmark for **cardinality-constrained mean–variance portfolio selection** formulated as a **QUBO**, evaluated with **depth-one QAOA (p=1)** on a classical statevector simulator, and compared to an **exact feasible-enumeration baseline** on small problem sizes.

The implementation is intentionally lightweight and uses only **Python + NumPy/Pandas/Matplotlib**.

---

## Repository structure

- `scripts/reproduce.py` — generates all tables/figures and CSV outputs
- `results/` — reference CSV outputs (from the paper)
- `figures/` — location for generated figures (recommended)
- `paper/` — LaTeX manuscript files (add `main.tex` and bibliography here)
- `.github/workflows/ci.yml` — GitHub Actions workflow that runs `reproduce.py`

---

## Quick start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Reproduce all outputs

```bash
python scripts/reproduce.py --outdir artifacts
```

This command writes:

- `results_instances.csv`
- `results_summary.csv`
- `results_ablation_topM.csv`
- `results_alpha_sensitivity.csv`
- `fig_gap_boxplot.pdf`

into the `artifacts/` directory.

---

## Notes on methodology (high level)

- **QUBO energy**: equal-weight utility proxy + quadratic penalty for fixed cardinality.
- **QAOA**: depth-one statevector simulation with a `(gamma, beta)` grid search that minimizes **expected energy**.
- **Reported portfolio**: hybrid **top-M** post-processing selects the minimum-energy candidate among the `M` most probable outcomes.

---

## Turkish kısa açıklama

Bu depo, kardinalite kısıtlı mean–variance portföy **seçim** problemini QUBO olarak kurar ve p=1 QAOA ile (statevector simülasyon) çalıştırır. Tüm tablo/figür çıktıları `scripts/reproduce.py` ile tek komutla yeniden üretilebilir.

---

## License

MIT — see `LICENSE`.

