import argparse
import locale
import os
import re
import subprocess
import sys
from datetime import datetime
from io import StringIO

import pandas as pd

locale.setlocale(locale.LC_NUMERIC, "C")

parser = argparse.ArgumentParser(
    description="Profile CUDA kernels using Nsight Compute."
)
parser.add_argument("executable", help="Path to the executable file.")
parser.add_argument("output_dir", help="Directory to save results.")
parser.add_argument(
    "-m", "--shared_memory", action="store_true", help="Enable shared memory."
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
shared_memory_flag = args.shared_memory
save_mode = args.save_mode

if not os.path.isfile(executable):
    print(f"Ошибка: файл '{executable}' не существует.")
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
elif not os.path.isdir(output_dir):
    print(f"Ошибка: '{output_dir}' не является директорией.")
    sys.exit(1)

filters = ["Sobel", "SobelSep", "Prewitt", "PrewittSep"]
sizes = [
    # "10x10",
    # "100x100",
    # "500x500",
    "1000x1000",
    # "2000x2000",
    # "3000x3000",
    # "4000x4000",
    # "5000x5000",
    # "10000x10000",
]
counts = [
    # "1", "2", "5", "10", "50", "100",
    "1000"
]
streams = [
    "",
    # "-a", "-a -o"
]
iterations = len(filters) * len(sizes) * len(counts) * len(streams)

output_excel = os.path.join(
    output_dir,
    f"ncu_profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
)


def convert_to_ms(value, unit):
    """Преобразование времени в миллисекунды."""
    try:
        value = float(value)
        if unit == "ns":
            return value / 1_000_000  # наносекунды в миллисекунды
        elif unit == "us":
            return value / 1_000  # микросекунды в миллисекунды
        elif unit == "ms":
            return value  # уже в миллисекундах
        elif unit == "s":
            return value * 1_000  # секунды в миллисекунды
        else:
            raise ValueError(f"Неизвестная единица измерения: {unit}")
    except (ValueError, TypeError) as e:
        print(f"Ошибка конвертации времени: {value} ({unit}) - {e}")
        return 0.0


def preprocess_metric_value(value):
    """Предобработка строки Metric Value для унификации формата чисел."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if re.match(r"^\d+,\d+$", value):
        value = value.replace(",", ".")
    elif re.match(r"^\d{1,3}(,\d{3})*(\.\d+)?$", value):
        value = value.replace(",", "")
    return value


def parse_ncu_output(output, f, s, c):
    lines = output.split("\n")
    csv_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('"ID"'):
            csv_start = i
            break
    if csv_start is None:
        print("CSV-данные не найдены в выводе.")
        return None

    csv_data = "\n".join(lines[csv_start:])
    try:
        df = pd.read_csv(StringIO(csv_data), thousands=",", decimal=".")
        df["Metric Value"] = df["Metric Value"].apply(preprocess_metric_value)
        df["Metric Value"] = pd.to_numeric(df["Metric Value"], errors="coerce")

        df["Function"] = f
        df["Size"] = s
        df["Count"] = c

        duration_rows = df[
            (df["Section Name"] == "Command line profiler metrics")
            & (df["Metric Name"] == "gpu__time_duration.sum")
        ]

        total_time = 0.0
        if not duration_rows.empty:
            total_time = duration_rows["Metric Value"].sum()
            unit = duration_rows["Metric Unit"].iloc[0]
            if pd.notna(total_time):
                total_time = convert_to_ms(total_time, unit)
                print(f"Суммарное время: {total_time} ms для {f}, {s}, {c}")
            else:
                print(f"Некорректное значение времени: {total_time} для {f}, {s}, {c}")
        else:
            print(f"Метрика 'gpu__time_duration.sum' не найдена для {f}, {s}, {c}")

        total_row = pd.DataFrame(
            {
                "Function": [f],
                "Size": [s],
                "Count": [c],
                "Metric Name": ["Total Duration"],
                "Metric Unit": ["ms"],
                "Metric Value": [total_time],
            }
        )
        df = pd.concat([df, total_row], ignore_index=True)
        return df
    except Exception as e:
        print(f"Ошибка парсинга CSV: {e}")
        return None


all_dfs = []

iteration = 0
for f in filters:
    for s in sizes:
        for c in counts:
            for st in streams:
                iteration += 1
                command_prefix = (
                    "LC_NUMERIC=C " if sys.platform.startswith("linux") else ""
                )
                command = f"{command_prefix}ncu --csv --metrics gpu__time_duration.sum {executable} -f {f} -s {s} -c {c} {st}"
                if shared_memory_flag:
                    command += " -m"
                print(f"[{iteration}/{iterations}] Выполняется: {command}")
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True
                    )
                    output = result.stdout + result.stderr
                    df = parse_ncu_output(output, f, s, c)
                    if df is not None and not df.empty:
                        cols = ["Function", "Size", "Count"] + [
                            col
                            for col in df.columns
                            if col not in ["Function", "Size", "Count"]
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
                        print(f"Нет данных для команды: {command}")
                except Exception as e:
                    print(f"Ошибка при выполнении команды '{command}': {e}")

if save_mode == "single" and all_dfs:
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for df, sheet_name in all_dfs:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Результаты сохранены в {output_excel}")
