#!/usr/bin/env python3
"""
demo_paper_table17.py — reconstructs cl_results.json from the paper's
Table 17 (15-task per-task evaluation matrix, Llama-3.2-1B-Instruct) and
Table 8b (routing assignments), then runs the intra/cross decomposition.

This gives the option-(iii) paper numbers immediately, without any GPU:
    python demo_paper_table17.py
Writes demo_run/cl_results.json and prints the full decomposition.
"""

import json
import os
import subprocess
import sys

ORDER = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "dbpedia", "amazon",
         "yahoo", "agnews", "yelp", "BoolQA", "QQP"]

# Table 8b (15-task stream routing)
TASK_TO_GROUP = {
    "C-STANCE": 0, "FOMC": 1, "MeetingBank": 2, "Py150": 3,
    "ScienceQA": 3, "NumGLUE-cm": 1, "NumGLUE-ds": 1, "20Minuten": 0,
    "dbpedia": 0, "amazon": 3, "yahoo": 0, "agnews": 0, "yelp": 0,
    "BoolQA": 0, "QQP": 3,
}

# Table 17, lower-triangular per-task evaluation matrix (percent)
ROWS = [
    [39.0],
    [39.0, 57.0],
    [39.0, 57.0, 19.3],
    [39.0, 57.0, 19.3, 37.9],
    [39.0, 57.0, 19.3, 37.2, 61.0],
    [39.0, 49.0, 19.3, 37.2, 61.0, 34.2],
    [39.0, 51.0, 19.3, 37.2, 61.0, 34.2, 41.0],
    [41.0, 51.0, 19.3, 37.2, 61.0, 34.2, 41.0, 38.4],
    [38.0, 51.0, 19.3, 37.2, 61.0, 34.2, 41.0, 38.2, 79.0],
    [38.0, 51.0, 19.3, 43.4, 57.0, 34.2, 41.0, 38.2, 79.0, 52.0],
    [40.0, 51.0, 19.3, 43.4, 57.0, 34.2, 41.0, 37.6, 80.0, 52.0, 61.0],
    [41.0, 51.0, 19.3, 43.4, 57.0, 34.2, 41.0, 38.1, 74.0, 52.0, 45.0,
     88.0],
    [37.0, 51.0, 19.3, 43.4, 57.0, 34.2, 41.0, 36.5, 72.0, 52.0, 44.0,
     85.0, 50.0],
    [40.0, 51.0, 19.3, 43.4, 57.0, 34.2, 41.0, 37.8, 77.0, 52.0, 42.0,
     82.0, 46.0, 61.0],
    [40.0, 51.0, 19.3, 36.4, 60.0, 34.2, 41.0, 37.8, 77.0, 52.0, 42.0,
     82.0, 46.0, 61.0, 75.0],
]


def main():
    n = len(ORDER)
    pm = [[None] * n for _ in range(n)]
    for i, row in enumerate(ROWS):
        assert len(row) == i + 1, f"row {i} has {len(row)} entries"
        for j, v in enumerate(row):
            pm[i][j] = v

    diag = {ORDER[j]: pm[j][j] for j in range(n)}
    last = [pm[n - 1][j] for j in range(n)]
    op = sum(last) / n
    bwt = sum(pm[j][j] - last[j] for j in range(n)) / n

    res = {
        "method": "CRAFT (paper Table 17 reconstruction)",
        "perf_matrix": pm,
        "diagonal": diag,
        "final_row": {ORDER[j]: last[j] for j in range(n)},
        "task_to_group": TASK_TO_GROUP,
        "overall_performance": round(op, 4),
        "backward_transfer": round(bwt, 4),
        "n_groups": len(set(TASK_TO_GROUP.values())),
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "demo_run")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cl_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[demo] wrote {out_dir}/cl_results.json  "
          f"(OP={op:.2f}, BWT={bwt:.2f})")

    analyzer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "analyze_intra_group_forgetting.py")
    sys.exit(subprocess.call(
        [sys.executable, analyzer, "--run", out_dir,
         "--latex", os.path.join(out_dir, "intra_split_table.tex"),
         "--json", os.path.join(out_dir, "events.json")]))


if __name__ == "__main__":
    main()
