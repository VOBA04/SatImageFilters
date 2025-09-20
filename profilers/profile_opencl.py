import argparse
import os
import re
import subprocess
from datetime import datetime

import pandas as pd

parser = argparse.ArgumentParser(
    description="Profile OpenCL kernels using portable event timing."
)
parser.add_argument("executable", help="Path to the OpenCL benchmark executable.")
parser.add_argument("output_dir", help="Directory to save results.")
parser.add_argument(
    "-m", "--shared_memory", action="store_true", help="Enable local memory (3x3 only)."
)
parser.add_argument(
    "--gauss-only",
    action="store_true",
    help="Run only Gaussian filters (Gauss, GaussSep).",
)
parser.add_argument(
    "--gauss-size",
    type=int,
    default=3,
    help="Gaussian kernel size for Gauss/GaussSep (default: 3)",
)
parser.add_argument(
    "--gauss-sigma",
    type=float,
    default=0.0,
    help="Gaussian kernel sigma for Gauss/GaussSep (default: 0.0)",
)
parser.add_argument(
    "--save-mode",
    choices=["single", "iterative"],
    default="single",
    help="Save as one Excel file or per-iteration sheets",
)
args = parser.parse_args()

exe = args.executable
out_dir = args.output_dir
use_local = args.shared_memory
only_gauss = args.gauss_only
k_size = args.gauss_size
k_sigma = args.gauss_sigma
save_mode = args.save_mode

if not os.path.isfile(exe):
    raise SystemExit(f"Ошибка: файл '{exe}' не существует.")
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)
elif not os.path.isdir(out_dir):
    raise SystemExit(f"Ошибка: '{out_dir}' не является директорией.")

filters = (
    ["Gauss", "GaussSep"]
    if only_gauss
    else ["Sobel", "SobelSep", "Prewitt", "PrewittSep"]
)
sizes = ["1000x1000"]
counts = ["1000"]


def parse_output(text, f, s, c):
    rows = []
    in_block = False
    device_total = None
    wall_total = None
    try:
        count_int = int(c)
    except Exception:
        count_int = 0
    for line in text.splitlines():
        line = line.strip()
        if line == "OPENCL_PROFILE_BEGIN":
            in_block = True
            continue
        if line == "OPENCL_PROFILE_END":
            in_block = False
            break
        if not in_block:
            continue
        if line.startswith("KERNEL,"):
            # KERNEL,<name>,MS,<value>
            m = re.match(r"KERNEL,([^,]+),MS,([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                rows.append(
                    {
                        "Function": f,
                        "Size": s,
                        "Count": c,
                        "Kernel": m.group(1),
                        "Metric Name": "kernel_ms",
                        "Metric Unit": "ms",
                        "Metric Value": float(m.group(2)),
                    }
                )
            continue
        if line.startswith("TOTAL_DEVICE_MS,"):
            device_total = float(line.split(",", 1)[1])
        if line.startswith("TOTAL_WALL_MS,"):
            wall_total = float(line.split(",", 1)[1])
    df = pd.DataFrame(rows)
    # Per-kernel averages (per image)
    avg_rows = []
    if not df.empty and count_int > 0:
        for _, r in df.iterrows():
            if r.get("Metric Name") == "kernel_ms":
                avg_rows.append(
                    {
                        "Function": f,
                        "Size": s,
                        "Count": c,
                        "Kernel": r["Kernel"],
                        "Metric Name": "avg_ms",
                        "Metric Unit": "ms",
                        "Metric Value": float(r["Metric Value"]) / float(count_int),
                    }
                )

    # Rollups: totals and averages (device and wall)
    total_device = device_total if device_total is not None else 0.0
    total_wall = wall_total if wall_total is not None else 0.0
    avg_device = (total_device / float(count_int)) if count_int > 0 else 0.0
    avg_wall = (total_wall / float(count_int)) if count_int > 0 else 0.0

    total_df = pd.DataFrame(
        [
            {
                "Function": f,
                "Size": s,
                "Count": c,
                "Kernel": "TOTAL_DEVICE",
                "Metric Name": "total_ms",
                "Metric Unit": "ms",
                "Metric Value": total_device,
            },
            {
                "Function": f,
                "Size": s,
                "Count": c,
                "Kernel": "TOTAL_DEVICE",
                "Metric Name": "avg_ms",
                "Metric Unit": "ms",
                "Metric Value": avg_device,
            },
            {
                "Function": f,
                "Size": s,
                "Count": c,
                "Kernel": "TOTAL_WALL",
                "Metric Name": "total_ms",
                "Metric Unit": "ms",
                "Metric Value": total_wall,
            },
            {
                "Function": f,
                "Size": s,
                "Count": c,
                "Kernel": "TOTAL_WALL",
                "Metric Name": "avg_ms",
                "Metric Unit": "ms",
                "Metric Value": avg_wall,
            },
        ]
    )

    # Console summary
    print(f"Суммарное время ядра (device): {total_device} ms для {f}, {s}, {c}")
    print(f"Среднее время на изображение (device): {avg_device} ms")
    if df.empty:
        return total_df
    extra_df = pd.DataFrame(avg_rows) if avg_rows else pd.DataFrame()
    frames = [df]
    if not extra_df.empty:
        frames.append(extra_df)
    frames.append(total_df)
    return pd.concat(frames, ignore_index=True)


output_excel = os.path.join(
    out_dir, f"opencl_profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)
all_dfs = []
iterations = len(filters) * len(sizes) * len(counts)
iter_idx = 0

for f in filters:
    for s in sizes:
        for c in counts:
            iter_idx += 1
            cmd = f"{exe} -f {f} -s {s} -c {c}"
            if f in ("Gauss", "GaussSep"):
                cmd += f" --gauss_size {k_size} --gauss_sigma {k_sigma}"
            if use_local:
                cmd += " -m"
            print(f"[{iter_idx}/{iterations}] Выполняется: {cmd}")
            try:
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                text = res.stdout + res.stderr
                df = parse_output(text, f, s, c)
                if df is not None and not df.empty:
                    cols = [
                        "Function",
                        "Size",
                        "Count",
                        "Kernel",
                        "Metric Name",
                        "Metric Unit",
                        "Metric Value",
                    ]
                    df = df[cols]
                    if save_mode == "iterative":
                        with pd.ExcelWriter(
                            output_excel,
                            engine="openpyxl",
                            mode="a" if os.path.exists(output_excel) else "w",
                        ) as writer:
                            sheet_name = f"{f}_{s}_{c}".replace(".", "_")[:31]
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        all_dfs.append((df, f"{f}_{s}_{c}".replace(".", "_")[:31]))
                else:
                    print(f"Нет данных для команды: {cmd}")
            except Exception as e:
                print(f"Ошибка при выполнении команды '{cmd}': {e}")

if save_mode == "single" and all_dfs:
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for df, sheet_name in all_dfs:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Результаты сохранены в {output_excel}")
