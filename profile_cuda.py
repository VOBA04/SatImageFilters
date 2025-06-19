import os
import re
import subprocess
import sys
from datetime import datetime
from io import StringIO

import pandas as pd
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

if len(sys.argv) != 3:
    print(
        "Ошибка: укажите путь к исполняемому файлу и директорию для сохранения результатов."
    )
    print(
        "Пример: python profile_cuda.py /path/to/benchmark_gpu.exe /path/to/output/dir"
    )
    sys.exit(1)

executable = sys.argv[1]
output_dir = sys.argv[2]

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
    f"cuda_profiling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
)


def convert_to_ms(time_value, unit):
    try:
        value = float(time_value)
        if unit == "ms":
            return value
        elif unit == "us":
            return value / 1000.0
        elif unit == "s":
            return value * 1000.0
        else:
            print(f"Неизвестная единица измерения: {unit}. Предполагается ms.")
            return value
    except ValueError:
        print(f"Ошибка конвертации времени: {time_value}")
        return 0.0


def parse_nvprof_output(output, f, s, c):
    match = re.search(r"==\d+== Profiling result:", output)
    if not match:
        print("Результаты профилирования не найдены в выводе.")
        return None, None
    csv_data = output[match.end() :].strip()
    df = pd.read_csv(StringIO(csv_data), header=0)
    units = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    new_columns = []
    for col, unit in zip(df.columns, units):
        if pd.notna(unit) and unit != "":
            new_columns.append(f"{col} ({unit})")
        else:
            new_columns.append(col)
    df.columns = new_columns
    df["Function"] = f
    df["Size"] = s
    df["Count"] = c
    time_col = [col for col in df.columns if col.startswith("Time (")][0]
    unit = time_col.split("(")[1].strip(")")
    total_time = df[time_col].apply(lambda x: convert_to_ms(x, unit)).sum()
    total_row = pd.DataFrame(
        {
            "Function": [f],
            "Size": [s],
            "Count": [c],
            time_col: [total_time],
            "Name": ["Total"],
        }
    )
    df = pd.concat([df, total_row], ignore_index=True)
    return df, total_time


with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    workbook = writer.book
    iteration = 0
    for f in filters:
        for s in sizes:
            total_times = []
            count_values = []
            for c in counts:
                iteration += 1
                command = f"nvprof --csv --trace gpu {executable} -f {f} -s {s} -c {c}"
                print(f"[{iteration}/{iterations}] Выполняется: {command}")
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True
                    )
                    output = result.stdout + result.stderr
                    df, total_time = parse_nvprof_output(output, f, s, c)
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
