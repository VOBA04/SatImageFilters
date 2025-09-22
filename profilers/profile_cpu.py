import argparse
import locale
import os
import re
import subprocess
import sys
from datetime import datetime

import pandas as pd

locale.setlocale(locale.LC_NUMERIC, "C")

parser = argparse.ArgumentParser(description="Profile CPU benchmark.")
parser.add_argument("executable", help="Path to the CPU benchmark executable.")
parser.add_argument("output_dir", help="Directory to save results.")
parser.add_argument(
    "--gauss-only",
    action="store_true",
    help="Run only Gaussian filters (Gauss, GaussSep).",
)
parser.add_argument(
    "--gauss-size",
    type=int,
    default=3,
    help="Gaussian kernel size to pass for Gauss/GaussSep (default: 3)",
)
parser.add_argument(
    "--gauss-sigma",
    type=float,
    default=1.0,
    help="Gaussian kernel sigma to pass for Gauss/GaussSep (default: 1.0)",
)
parser.add_argument(
    "--save-mode",
    choices=["single", "iterative"],
    default="single",
    help="Save mode: 'single' for one Excel file at the end, 'iterative' for saving each iteration.",
)
args = parser.parse_args()

executable = args.executable
output_dir = args.output_dir
gauss_only = args.gauss_only
gauss_size = args.gauss_size
gauss_sigma = args.gauss_sigma
save_mode = args.save_mode

if not os.path.isfile(executable):
    print(f"Ошибка: файл '{executable}' не существует.")
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
elif not os.path.isdir(output_dir):
    print(f"Ошибка: '{output_dir}' не является директорией.")
    sys.exit(1)

filters = (
    ["Gauss", "GaussSep"]
    if gauss_only
    else ["Sobel", "SobelSep", "Prewitt", "PrewittSep"]
)
sizes = [
    # "10x10",
    # "100x100",
    # "500x500",
    "1000x1000",
    # "2000x2000",
    # "3000x3000",
]
counts = [
    # "1", "2", "5",
    # "10",
    # "50",
    "100",
    # "1000"
]

iterations = len(filters) * len(sizes) * len(counts)

output_excel = os.path.join(
    output_dir,
    f"cpu_profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
)

total_pattern = re.compile(r"Total time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.I)
avg_pattern = re.compile(r"Average per image:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.I)
checksum_pattern = re.compile(r"Checksum:\s*([0-9]+)", re.I)


def parse_benchmark_output(output, f, s, c):
    total_ms = None
    avg_ms = None
    checksum = None

    for line in output.splitlines():
        if total_ms is None:
            m = total_pattern.search(line)
            if m:
                total_ms = float(m.group(1))
                continue
        if avg_ms is None:
            m = avg_pattern.search(line)
            if m:
                avg_ms = float(m.group(1))
                continue
        if checksum is None:
            m = checksum_pattern.search(line)
            if m:
                checksum = int(m.group(1))
                continue

    if total_ms is None and avg_ms is None:
        print("Не удалось распарсить время из вывода бенчмарка.")
        return None

    return {
        "Function": f,
        "Size": s,
        "Count": c,
        "Total (ms)": total_ms if total_ms is not None else (float(avg_ms) * float(c)),
        "Average (ms)": avg_ms if avg_ms is not None else (float(total_ms) / float(c)),
        "Checksum": checksum,
    }


all_rows = []
iteration = 0

for f in filters:
    for s in sizes:
        for c in counts:
            iteration += 1
            command_prefix = "LC_NUMERIC=C " if sys.platform.startswith("linux") else ""
            cmd = f"{command_prefix}{executable} -f {f} -s {s} -c {c}"
            if f in ("Gauss", "GaussSep"):
                cmd += f" --gauss_size {gauss_size} --gauss_sigma {gauss_sigma}"
            print(f"[{iteration}/{iterations}] Выполняется: {cmd}")
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout + result.stderr
                row = parse_benchmark_output(output, f, s, c)
                if row:
                    try:
                        t_ms = float(row.get("Total (ms)", "nan"))
                        a_ms = float(row.get("Average (ms)", "nan"))
                        checksum = row.get("Checksum")
                        print(
                            f"  -> Total time: {t_ms:.3f} ms | Average per image: {a_ms:.3f} ms"
                            + (
                                f" | Checksum: {checksum}"
                                if checksum is not None
                                else ""
                            )
                        )
                    except Exception:
                        pass
                    df = pd.DataFrame([row])
                    if save_mode == "iterative":
                        with pd.ExcelWriter(
                            output_excel,
                            engine="openpyxl",
                            mode="a" if os.path.exists(output_excel) else "w",
                        ) as writer:
                            sheet_name = f"{f}_{s}_{c}".replace(".", "_")[:31]
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        all_rows.append(row)
                else:
                    print(f"Нет данных для команды: {cmd}")
            except Exception as e:
                print(f"Ошибка при выполнении команды '{cmd}': {e}")

if save_mode == "single" and all_rows:
    df = pd.DataFrame(all_rows)
    cols = ["Function", "Size", "Count", "Total (ms)", "Average (ms)", "Checksum"]
    df = df[[c for c in cols if c in df.columns]]
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)

print(f"Результаты сохранены в {output_excel}")
