# SatImageFilters

![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/VOBA04/7f58e72739aa381f9225edc7315e3d72/raw/coverage.json)
[![wakatime](https://wakatime.com/badge/user/11288404-d9dc-4a59-abbb-f91d607d51fc/project/828a01b2-3c50-45e3-954e-c916bc5026bc.svg)](https://wakatime.com/badge/user/11288404-d9dc-4a59-abbb-f91d607d51fc/project/828a01b2-3c50-45e3-954e-c916bc5026bc)

Учебно‑прикладной проект по обработке TIFF‑изображений с реализацией операторов Собеля и Превитта, а также гауссова размытия (в том числе разделённой свёртки) на CPU, CUDA и OpenCL. Есть консольные утилиты, модульные тесты и (опционально) GUI на Qt6 (если установлен; часть функций использует CUDA при её наличии).

## Возможности

- Чтение/запись TIFF через libtiff (`TIFFImage`).
- Класс ядер свёртки и генерация гауссова ядра (`Kernel`).
- Фильтры: Собель, Превитт, Гаусс (обычный и разделённый).
- Исполнения: CPU, CUDA (при `BUILD_WITH_CUDA=ON`), OpenCL.
- Профилирование OpenCL (сбор метрик выполнения ядер).
- Тесты на GoogleTest, бенчмарки CPU/CUDA/OpenCL.
- GUI на Qt6 (собирается при наличии Qt6; функции CUDA активны при включённой поддержке).

## Зависимости и версии (собираются автоматически при отсутствии в системе)

- C++17, CMake ≥ 3.20
- libtiff 4.7.0
- OpenCV 4.10.0 (используется для GUI/захвата видео и тестирования)
- GoogleTest 1.14.0
- Doxygen 1.9.x (для документации)
- Опционально:
  - CUDA Toolkit (автоопределение версии или точная через `-DCUDA_VERSION=`)
  - Qt6 (Core/Gui/Widgets) — для GUI
  - OpenCL (Headers + ICD Loader) — если нет в системе, соберётся автоматически

Все внешние зависимости кэшируются в `external_build/<OS-ARCH>`, `CMAKE_PREFIX_PATH` указывает туда же.

## Клонирование

```bash
git clone <URL_репозитория>
cd SatImageFilters
git submodule update --init --recursive
```

## Сборка

Рекомендуемый универсальный способ (Linux/macOS/WSL):

```bash
cmake -S . -B build
cmake --build build -j
```

Параметры CMake:

- `-DBUILD_WITH_CUDA=ON|OFF` — собрать CUDA-часть (по умолчанию ON).
- `-DCUDA_VERSION=AUTO|<x.y>` — автоматически искать Toolkit или требовать точную версию.
- `-DCUDA_ARCH=AUTO|<list>` — архитектуры для NVCC (по умолчанию авто/предустановки).

Примеры платформенных генераторов:

- Linux (Unix Makefiles или Ninja):
  
  ```bash
  cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build build -j
  ```
  
- Windows (Visual Studio 2019):
  
  ```powershell
  cmake -S . -B build -G "Visual Studio 16 2019" -A x64
  cmake --build build --config Release
  ```

После сборки исполняемые файлы находятся в `build/` (на Windows — в подкаталогах конфигурации, например `build/Release`).

## Собираемые цели (основные)

- `SatImageFilters` — пример запуска фильтров (CUDA при наличии).
- `unit_tests` — модульные тесты (GoogleTest).
- `benchmark_cpu`, `benchmark_opencl` — бенчмарки CPU и OpenCL.
- `benchmark_gpu` — бенчмарк CUDA (если `BUILD_WITH_CUDA=ON`).
- `speedtest` — тест производительности CUDA (при `BUILD_WITH_CUDA=ON`).
- `main_opencl` — пример запуска OpenCL.
- `gui` — графический интерфейс (Qt6 + CUDA; собирается при наличии Qt6 и `BUILD_WITH_CUDA=ON`).

Файл `kernels.cl` копируется в билд‑директорию автоматической целью CMake для OpenCL‑примеров.

## Запуск

- Тесты: можно запустить `unit_tests` напрямую; также тест регистрируется в CTest при `BUILD_WITH_CUDA=ON`.
- CLI‑примеры и бенчмарки принимают параметры командной строки (парсер `CommandLineParser`).
- GUI показывает базовый pipeline: загрузка изображения, выбор операции, параметры гауссова размытия, сохранение результата, а также работу с видеопотоком (через OpenCV).

## Скрипты профилирования (Python)

В папке `profilers/` находятся скрипты для автоматизации профилирования CPU/CUDA/OpenCL-бенчмарков и выгрузки результатов в Excel (`.xlsx`).

- Требования:
  - Python 3.8+ (рекомендовано 3.10+)
  - Пакеты: `pandas`, `openpyxl`
  - Для CUDA Nsight Compute: установлен `ncu` (входит в CUDA Toolkit)
  - Для nvprof (устаревший): установлен `nvprof`

Установка зависимостей:

```bash
python3 -m pip install -U pandas openpyxl
```

Общие флаги для скриптов:

- `--gauss-only` — опционально; профилируются только фильтры Гаусса (`Gauss`, `GaussSep`). Без этого флага запускаются операторы Собеля/Превитта (и их разделённые варианты: `SobelSep`, `PrewittSep`).
- `-m`/`--shared_memory` — опционально; тест операций с разделяемой (CUDA) / локальной (OpenCL) памятью. Ограничение: для OpenCL актуально для ядра 3×3.
- `--save-mode {single|iterative}` — режим сохранения результатов (по умолчанию `single`).
  - `single`: сохранить один Excel‑файл в конце выполнения. Для CPU — на лист `results`; для OpenCL/CUDA‑NCU — на отдельные вкладки по конфигурациям.
  - `iterative`: сохранять/добавлять результаты по мере выполнения, создавая/дополняя вкладку для каждой конфигурации. Удобно, если запуск долгий или может быть прерван.
  - Примечание: `profile_cuda_nvprof.py` не поддерживает `--save-mode` и всегда пишет один файл с вкладками по конфигурациям, дополнительно строя графики.

Скрипты и примеры запуска:

- CPU: `profilers/profile_cpu.py`
  - Запускает `benchmark_cpu`, парсит суммарное и среднее время, пишет `cpu_profiling_results_*.xlsx`.
  
  ```bash
  python3 profilers/profile_cpu.py build/benchmark_cpu dump/ \
    --gauss-only --gauss-size 3 --gauss-sigma 1.0 --save-mode single
  ```

- OpenCL: `profilers/profile_opencl.py`
  - Запускает `benchmark_opencl`, собирает метрики из консольного профайлера (по маркерам OPENCL_PROFILE_*), сохраняет `opencl_profiling_results_*.xlsx`.
  - Флаг `-m/--shared_memory` включает локальную память (только 3x3).
  
  ```bash
  python3 profilers/profile_opencl.py build/benchmark_opencl dump/ \
    -m --gauss-only --gauss-size 3 --gauss-sigma 1.0 --save-mode single
  ```

- CUDA (Nsight Compute): `profilers/profile_cuda_ncu.py`
  - Использует `ncu --csv` и метрику `gpu__time_duration.sum`, сохраняет `ncu_profiling_results_*.xlsx`.
  - Флаг `-m/--shared_memory` передаётся в бенчмарк.
  
  ```bash
  python3 profilers/profile_cuda_ncu.py build/benchmark_gpu dump/ \
    -m --gauss-only --gauss-size 3 --gauss-sigma 1.0 --save-mode single
  ```

- CUDA (nvprof, устаревший): `profilers/profile_cuda_nvprof.py`
  - Использует `nvprof --csv`, сохраняет листы по конфигурациям и строит графики в Excel, файл `cuda_profiling_results_*.xlsx`.
  - Флаг `-m/--shared_memory` передаётся в бенчмарк.
  
  ```bash
  python3 profilers/profile_cuda_nvprof.py build/benchmark_gpu dump/ -m
  ```

По умолчанию скрипты используют наборы размеров и повторов, заданные внутри файлов, и помещают результаты в указанную директорию (например, `dump/`).

## Документация (Doxygen)

Сгенерируйте документацию:

```bash
doxygen Doxyfile
```

Откройте `doxygen/html/index.html` в браузере. Главная страница — краткий обзор проекта, затем переходите к классам `TIFFImage`, `Kernel`, `CudaMemManager`, `MainWindow` и др.

## Структура данных и папок

- `include/` — заголовки публичного API (в т.ч. `gui/`).
- `src/` — реализации, примеры (`mains/`), GUI (`gui/`), OpenCL‑ядра (`kernels/`).
- `tests/` — модульные тесты.
- `images/` — рабочая директория с результатами.
  - `original/`, `prewitt/`, `sobel/`, `gaussian/`, `arbitrary_kernel/`, `speedtest/` — будут созданы автоматически при первом запуске.
- `profilers/` — Python‑скрипты профилирования для CPU/CUDA/OpenCL.
- `external/`, `external_build/` — исходники и артефакты сторонних зависимостей.
- `doxygen/` — выходная директория документации.

### Файл kernel.txt

Для произвольного ядра свёртки используйте `kernel.txt` в корне проекта:

1) Размер ядра (высота ширина — нечётные числа)
2) Флаг «можно ли вращать ядро» (0/1)
3) Значения ядра построчно

Пример:

```text
3 3
1
-1 0 1
-2 0 2
-1 0 1
```

## Примечания

- GUI цель собирается при наличии Qt6; использование CUDA доступно при включённой поддержке.
- Если OpenCL отсутствует в системе, CMake соберёт необходимые заголовки и ICD‑лоадер в `external_build/` и подключит их автоматически.
- Для CUDA версий и архитектур предусмотрены параметры `CUDA_VERSION` и `CUDA_ARCH` (см. раздел «Сборка»).
