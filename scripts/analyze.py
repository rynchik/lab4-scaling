"""Таблицы, графики и описательная модель Амдала по сохранённым измерениям."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "matvec_sendrecv": "A @ x: Send/Recv",
    "matvec_scatterv": "A @ x: Scatterv",
    "matvec_cached": "A @ x: кэш матрицы",
    "dot_reduce": "Скалярное: Reduce",
    "dot_allreduce": "Скалярное: Allreduce",
    "transpose": "A.T @ b: Reduce",
    "batch10_baseline": "10 умножений: повторная раздача",
    "batch10_cached": "10 умножений: кэш",
    "cg_full": "CG: полная",
    "cg_full_typed": "CG: полная, буферы",
    "cg_simple": "CG: упрощённая",
}
CG = ["cg_full", "cg_full_typed", "cg_simple"]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "report/local",
    )
    args = parser.parse_args()
    target = args.input
    plan = json.loads((target / "plan.json").read_text(encoding="utf-8"))
    rows, micro, profiles = [], [], []
    raw_jobs = []
    for job in plan["jobs"]:
        path = (
            target / "raw" / f'{job["scaling"]}_{job["name"]}_p{job["processes"]}.json'
        )
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["M"], data["N"], data["processes"]) != (
            job["M"],
            job["N"],
            job["processes"],
        ):
            raise ValueError(f"Параметры результата не соответствуют плану: {path}")
        raw_jobs.append((job, data))
        micro.extend(data["microbench"])
        for result in data["rows"]:
            rows.append(
                {
                    "profile": plan["profile"],
                    "scaling": job["scaling"],
                    "case": job["name"],
                    "M": data["M"],
                    "N": data["N"],
                    "processes": data["processes"],
                    "method": result["method"],
                    "median_s": result["median_s"],
                    "min_s": min(result["samples"]),
                    "max_s": max(result["samples"]),
                    "speedup": None,
                    "efficiency": None,
                    "iterations": result.get("iterations", [""])[0],
                    "relative_error": result["max_relative_error"],
                    "scope": result["scope"],
                    "oversubscribed": (
                        data["processes"] > (plan.get("logical_cpus") or 10**9)
                        if plan["profile"] == "local"
                        else "not_assessed"
                    ),
                }
            )
            if result.get("phase_samples"):
                for phase in result["phase_samples"][0]:
                    profiles.append(
                        {
                            "case": job["name"],
                            "scaling": job["scaling"],
                            "method": result["method"],
                            "processes": data["processes"],
                            "phase": phase,
                            "median_max_rank_s": float(
                                np.median([s[phase] for s in result["phase_samples"]])
                            ),
                            "calls_per_solve": result["calls_per_solve"][phase],
                        }
                    )
    if not rows:
        raise SystemExit("Измерений нет; сначала запустите run_suite.py")
    for row in rows:
        baseline = next(
            r["median_s"]
            for r in rows
            if (r["scaling"], r["case"], r["method"], r["processes"])
            == (row["scaling"], row["case"], row["method"], 1)
        )
        ratio = baseline / row["median_s"]
        if row["scaling"] == "strong":
            row["speedup"], row["efficiency"] = ratio, ratio / row["processes"]
        else:
            row["speedup"], row["efficiency"] = row["processes"] * ratio, ratio
    write_csv(target / "metrics.csv", rows)
    write_csv(target / "communication_profile.csv", profiles)
    write_csv(
        target / "microbench.csv",
        [
            {key: row[key] for key in ("operation", "N", "processes", "median_s")}
            for row in micro
        ],
    )
    strong = [r for r in rows if r["scaling"] == "strong"]
    weak = [r for r in rows if r["scaling"] == "weak"]
    cases = list(dict.fromkeys(row["case"] for row in strong))

    def select(case, method, data=strong):
        return sorted(
            [r for r in data if r["case"] == case and r["method"] == method],
            key=lambda r: r["processes"],
        )

    fig, axes = plt.subplots(
        len(cases), 3, figsize=(13, 3.5 * len(cases)), squeeze=False
    )
    for index, case in enumerate(cases):
        for method in CG:
            selected = select(case, method)
            p = [r["processes"] for r in selected]
            for col, key, multiplier in [
                (0, "median_s", 1000),
                (1, "speedup", 1),
                (2, "efficiency", 100),
            ]:
                axes[index, col].plot(
                    p,
                    [r[key] * multiplier for r in selected],
                    "o-",
                    label=LABELS[method],
                )
        for col, label in enumerate(
            ["Время цикла, мс", "Ускорение T(1)/T(p)", "Эффективность, %"]
        ):
            ax = axes[index, col]
            ax.set_title(f"Набор {case}")
            ax.set(xlabel="Число MPI-процессов", ylabel=label)
            ax.set_xticks(p)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    profile_label = (
        "уменьшенные наборы" if plan["profile"] == "local" else "исходные наборы"
    )
    fig.suptitle(f"Сильное масштабирование CG; {profile_label}")
    fig.tight_layout()
    fig.savefig(target / "strong_cg.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(cases), figsize=(13, 7), squeeze=False)
    for col, case in enumerate(cases):
        for index, methods in enumerate(
            [
                ["matvec_sendrecv", "matvec_scatterv", "matvec_cached"],
                ["dot_reduce", "dot_allreduce", "transpose"],
            ]
        ):
            ax = axes[index, col]
            for method in methods:
                selected = select(case, method)
                ax.semilogy(
                    [r["processes"] for r in selected],
                    [r["median_s"] * 1000 for r in selected],
                    "o-",
                    label=LABELS[method],
                )
            ax.set(
                title=f"Набор {case}, ЛР{index+1}",
                xlabel="Число MPI-процессов",
                ylabel="Время, мс (лог. шкала)",
            )
            ax.set_xticks(p)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Матрично-векторные операции; для кэша раздача вынесена из вызова")
    fig.tight_layout()
    fig.savefig(target / "strong_operations.png", dpi=150)
    plt.close(fig)

    if weak:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in CG:
            selected = select("weak", method, weak)
            p = [r["processes"] for r in selected]
            axes[0].plot(
                p, [r["median_s"] * 1000 for r in selected], "o-", label=LABELS[method]
            )
            axes[1].plot(
                p, [r["efficiency"] * 100 for r in selected], "o-", label=LABELS[method]
            )
        for ax in axes:
            ax.set_xlabel("Число MPI-процессов")
            ax.set_xticks(p)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Время 20 итераций, мс")
        axes[1].set_ylabel("Слабая эффективность T(1)/T(p), %")
        fig.suptitle(
            f"Слабое масштабирование: {plan['config']['weak']['rows_per_rank']} строк на ранг, N=200"
        )
        fig.tight_layout()
        fig.savefig(target / "weak.png", dpi=150)
        plt.close(fig)

    models = []
    fig, axes = plt.subplots(1, len(cases), figsize=(12, 4), squeeze=False)
    for col, case in enumerate(cases):
        selected = select(case, "cg_simple")
        fit_rows = [r for r in selected if 1 < r["processes"] <= 4]
        a = np.array([1.0 - 1.0 / r["processes"] for r in fit_rows])
        y = np.array([1.0 / r["speedup"] - 1.0 / r["processes"] for r in fit_rows])
        raw_fraction = float(a @ y / (a @ a))
        fraction = float(np.clip(raw_fraction, 0.0, 1.0))
        p = np.array([r["processes"] for r in selected])
        prediction = 1.0 / (fraction + (1.0 - fraction) / p)
        rmse = float(
            np.sqrt(np.mean((prediction - [r["speedup"] for r in selected]) ** 2))
        )
        models.append(
            {
                "case": case,
                "method": "cg_simple",
                "fit_processes": "2,4",
                "raw_f": raw_fraction,
                "bounded_f": fraction,
                "speedup_rmse_all_p": rmse,
            }
        )
        ax = axes[0, col]
        ax.plot(p, [r["speedup"] for r in selected], "o-", label="Измерено")
        ax.plot(p, prediction, "--", label=f"Модель Амдала, f={fraction:.3f}")
        ax.set(
            title=f"Набор {case}",
            xlabel="Число MPI-процессов",
            ylabel="Ускорение CG, упрощённая",
        )
        ax.set_xticks(p)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target / "amdahl.png", dpi=150)
    plt.close(fig)
    write_csv(target / "amdahl.csv", models)

    optimized = []
    for case in cases:
        for p in sorted({r["processes"] for r in strong}):
            lookup = {
                r["method"]: r
                for r in strong
                if r["case"] == case and r["processes"] == p
            }
            optimized.append(
                {
                    "case": case,
                    "processes": p,
                    "cg_object_s": lookup["cg_full"]["median_s"],
                    "cg_typed_s": lookup["cg_full_typed"]["median_s"],
                    "cg_typed_gain": lookup["cg_full"]["median_s"]
                    / lookup["cg_full_typed"]["median_s"],
                    "batch10_baseline_s": lookup["batch10_baseline"]["median_s"],
                    "batch10_cached_s": lookup["batch10_cached"]["median_s"],
                    "batch10_gain": lookup["batch10_baseline"]["median_s"]
                    / lookup["batch10_cached"]["median_s"],
                }
            )
    write_csv(target / "optimizations.csv", optimized)
    print(
        f"Analyzed {len(raw_jobs)} jobs, {len(rows)} method/size/process combinations"
    )


if __name__ == "__main__":
    main()
