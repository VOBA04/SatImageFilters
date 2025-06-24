import locale
import os
import re
import subprocess
import sys
from datetime import datetime
from io import StringIO

import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

locale.setlocale(locale.LC_NUMERIC, "C")

shared_memory_flag = False
args = sys.argv[1:]
if "-m" in args:
    shared_memory_flag = True
    args.remove("-m")
elif "--shared_memory" in args:
    shared_memory_flag = True
    args.remove("--shared_memory")

if len(args) != 2:
    print(
        "Ошибка: укажите путь к исполняемому файлу и директорию для сохранения результатов."
    )
    print(
        "Пример: python profile_cuda_ncu.py /path/to/benchmark_gpu /path/to/output/dir [-m|--shared_memory]"
    )
    sys.exit(1)

executable = args[0]
output_dir = args[1]

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
    "10x10",
    "100x100",
    "500x500",
    "1000x1000",
    "2000x2000",
    "3000x3000",
    "4000x4000",
    "5000x5000",
    "10000x10000",
]
counts = ["1", "2", "5", "10", "50", "100", "1000"]

iterations = len(filters) * len(sizes) * len(counts)

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
        return None, None

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
            total_time = duration_rows[
                "Metric Value"
            ].sum()  # Суммируем время всех запусков
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
        return df, total_time
    except Exception as e:
        print(f"Ошибка парсинга CSV: {e}")
        return None, None


with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    workbook = writer.book
    iteration = 0
    for f in filters:
        for s in sizes:
            total_times = []
            count_values = []
            for c in counts:
                iteration += 1
                command_prefix = (
                    "LC_NUMERIC=C " if sys.platform.startswith("linux") else ""
                )
                command = f"{command_prefix}ncu --csv --metrics gpu__time_duration.sum {executable} -f {f} -s {s} -c {c}"
                if shared_memory_flag:
                    command += " -m"
                print(f"[{iteration}/{iterations}] Выполняется: {command}")
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True
                    )
                    output = result.stdout + result.stderr
                    df, total_time = parse_ncu_output(output, f, s, c)
                    if df is not None and not df.empty:
                        cols = ["Function", "Size", "Count"] + [
                            col
                            for col in df.columns
                            if col not in ["Function", "Size", "Count"]
                        ]
                        df = df[cols]
                        sheet_name = f"{f}_{s}_{c}".replace(".", "_")
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        total_times.append(total_time)
                        count_values.append(int(c))
                    else:
                        print(f"Нет данных для команды: {command}")
                except Exception as e:
                    print(f"Ошибка при выполнении команды '{command}': {e}")
            if total_times:
                plot_df = pd.DataFrame(
                    {"Count": count_values, "Total Time (ms)": total_times}
                )
                plot_df = plot_df.sort_values(by="Count")
                plot_sheet_name = f"Plot_{f}_{s}".replace(".", "_")
                if len(plot_sheet_name) > 31:
                    plot_sheet_name = plot_sheet_name[:31]
                plot_sheet = workbook.create_sheet(plot_sheet_name)
                for r in dataframe_to_rows(plot_df, index=False, header=True):
                    plot_sheet.append(r)
                chart = LineChart()
                chart.title = f"Total Time vs Count for {f} {s}"
                chart.x_axis.title = "Count"
                chart.y_axis.title = "Total Time (ms)"
                data = Reference(
                    plot_sheet, min_col=2, min_row=1, max_row=len(plot_df) + 1
                )
                categories = Reference(
                    plot_sheet, min_col=1, min_row=2, max_row=len(plot_df) + 1
                )
                chart.add_data(data, titles_from_data=True)
                chart.set_categories(categories)
                plot_sheet.add_chart(chart, "E5")

print(f"Результаты сохранены в {output_excel}")
