"""Оптимизированная последовательная база NumPy без коммуникаций MPI."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load  # noqa: E402


def sequential_cg(A, b, tol=1e-9):
    x = np.zeros(A.shape[1])
    r = A.T @ b
    p = r.copy()
    rr = float(r @ r)
    threshold = max(1e-12, tol * np.sqrt(rr))
    start = time.perf_counter()
    count = 0
    for count in range(1, 4 * A.shape[1] + 1):
        q = A.T @ (A @ p)
        step = rr / float(p @ q)
        x += step * p
        r -= step * q
        new_rr = float(r @ r)
        if np.sqrt(new_rr) <= threshold:
            break
        p *= new_rr / rr
        p += r
        rr = new_rr
    return x, time.perf_counter() - start, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "report/local/serial_baseline.csv"
    )
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/local.json").read_text(encoding="utf-8"))
    rows = []
    for case in config["cases"]:
        A, b, x, gradient = load(args.data_root / "local" / case["name"])
        methods = [
            ("matvec", lambda: A @ x, b),
            ("dot", lambda: b @ b, b @ b),
            ("transpose", lambda: A.T @ b, gradient),
        ]
        for method, operation, reference in methods:
            samples = []
            for trial in range(4):
                start = time.perf_counter()
                value = operation()
                elapsed = time.perf_counter() - start
                np.testing.assert_allclose(value, reference, rtol=1e-10, atol=1e-12)
                if trial:
                    samples.append(elapsed)
            rows.append(
                {
                    "case": case["name"],
                    "method": method,
                    "median_s": float(np.median(samples)),
                    "iterations": "",
                    "samples": json.dumps(samples),
                }
            )
        samples = []
        for trial in range(4):
            value, elapsed, iterations = sequential_cg(A, b)
            np.testing.assert_allclose(value, x, rtol=1e-6, atol=1e-7)
            if trial:
                samples.append(elapsed)
        rows.append(
            {
                "case": case["name"],
                "method": "cg_numpy",
                "median_s": float(np.median(samples)),
                "iterations": iterations,
                "samples": json.dumps(samples),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Serial NumPy baseline saved")


if __name__ == "__main__":
    main()
