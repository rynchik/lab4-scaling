# Лабораторная работа №4

Исследование эффективности MPI-программ из ЛР1–ЛР3. Объяснения, результаты и ответы на вопросы: [report.md](report.md).

```bash
python -m pip install -r requirements.txt
python scripts/run_suite.py --profile local
python scripts/analyze.py --input report/local
```

Нужны Python и MPI; `mpiexec` должен быть в PATH (либо задайте MPIEXEC).
Замеры выполнены локально на уменьшенных наборах. Доступа к кластеру нет; исходные размеры и 32–64 процесса не измерены. При p=16 компьютер перегружен.

Полные измерения — `report/local/raw/`, сводные метрики — `report/local/metrics.csv`. Генерируемые матрицы в архив не включены; пример есть в `data/sample/`.
Для отдельной базы NumPy: `python scripts/serial_baseline.py --data-root data/generated`; ограничьте OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS значением 1.
Кластерная конфигурация и Slurm-сценарий подготовлены, но не запускались. Подробности — в отчёте.
