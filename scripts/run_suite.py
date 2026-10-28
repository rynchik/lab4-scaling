"""Запустить уменьшенное локальное исследование или исходный кластерный план."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["local", "cluster"], default="local")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--launcher", choices=["mpiexec", "srun"], default="mpiexec")
    parser.add_argument("--tasks-per-node", type=int, default=32)
    parser.add_argument("--processes", nargs="+", type=int)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(
        (ROOT / f"configs/{args.profile}.json").read_text(encoding="utf-8")
    )
    processes = args.processes or config["processes"]
    if any(p <= 0 for p in processes) or args.tasks_per_node <= 0:
        parser.error("Число процессов и tasks-per-node должны быть положительными")
    processes = sorted(set([1, *processes]))
    output = args.output or ROOT / "report" / args.profile
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
    )
    jobs = []
    for case in config["cases"]:
        for p in processes:
            jobs.append({**case, "processes": p, "scaling": "strong"})
    weak = config["weak"]
    for p in config.get("weak_processes", processes):
        if p in processes:
            jobs.append(
                {
                    "name": "weak",
                    "M": weak["rows_per_rank"] * p,
                    "N": weak["N"],
                    "seed": weak["seed"],
                    "processes": p,
                    "scaling": "weak",
                    "fixed_iters": weak["iterations"],
                }
            )
    plan = {
        "profile": args.profile,
        "config": config,
        "jobs": jobs,
        "execution_status": "not_run",
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.plan_only:
        print(f"Plan saved: {len(jobs)} jobs; no experiments executed")
        return
    if args.profile == "cluster" and not args.allow_large:
        parser.error(
            "Исходные данные занимают десятки ГБ; кластерный запуск требует --allow-large"
        )
    launcher = (
        (os.environ.get("MPIEXEC") or shutil.which("mpiexec"))
        if args.launcher == "mpiexec"
        else shutil.which("srun")
    )
    if not launcher:
        parser.error("Не найден MPI launcher")
    for job in jobs:
        p = job["processes"]
        case_dir = (
            args.data_root
            / args.profile
            / (job["name"] if job["scaling"] == "strong" else f"weak_p{p}")
        )
        manifest_file = case_dir / "manifest.json"
        expected = {key: job[key] for key in ("M", "N", "seed")}
        existing = (
            json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest_file.exists()
            else {}
        )
        if any(existing.get(key) != value for key, value in expected.items()):
            generate(case_dir, **expected)
        target = output / "raw" / f'{job["scaling"]}_{job["name"]}_p{p}.json'
        if args.launcher == "srun":
            prefix = [
                launcher,
                "--exclusive",
                "--exact",
                f"--nodes={(p + args.tasks_per_node - 1) // args.tasks_per_node}",
                f"--ntasks={p}",
                "--cpus-per-task=1",
                "--cpu-bind=cores",
            ]
        else:
            prefix = [launcher, "-n", str(p)]
        command = [
            *prefix,
            sys.executable,
            str(ROOT / "scripts/worker.py"),
            "--data",
            str(case_dir),
            "--output",
            str(target),
            "--repeats",
            str(config["repeats"]),
            "--warmup",
            str(config["warmup"]),
        ]
        if job.get("fixed_iters"):
            command += ["--fixed-iters", str(job["fixed_iters"])]
        if job["scaling"] == "strong" and job["name"] == "A":
            command += ["--microbench"]
        subprocess.run(command, check=True, env=env)
        job["status"] = "completed"
        plan["execution_status"] = "partial"
        (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    plan["execution_status"] = "completed"
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
