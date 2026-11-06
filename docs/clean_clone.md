# Запуск из нового клона

## Входные файлы

Для обучения нужны разрешённые train.csv, test.csv и sample_submit.csv из приватного соревнования.
Поместите их в data/raw/. Git не содержит эти CSV, сохранённые модели, контрольные артефакты или HTTP cache.

## Установка и обучение

Из корня новой папки с кодом в Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 6 -r requirements-app.txt -r requirements-reports.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/run.py train
.\.venv\Scripts\python.exe scripts/run.py predict
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --browser.gatherUsageStats=false
```

Команда train создаёт artifacts/baseline/model.joblib, который загружает демо.
Predict создаёт artifacts/submission.csv. В существующей папке эти команды заменяют соответствующие выходы;
для повторного обучения сохраняйте контрольную версию и задавайте отдельный output.

Команда reproduce создаёт artifacts/reproduced/model/model.joblib и artifacts/reproduced/submission.csv.
Она не переключает демо на другую модель. Для нового клона основной сценарий выше использует train.

Без модели и приватных входных файлов часть UI-тестов пропускается.
Полный набор следует запускать после train и predict.

## Контроль воспроизведения

Для сравнения с исходной версией нужен доверенный контрольный пакет, полученный отдельно от Git.
Сгенерированный заново CSV сам по себе не является таким пакетом.

```powershell
.\.venv\Scripts\python.exe scripts/run.py reproduce --output artifacts/reproduced
.\.venv\Scripts\python.exe scripts/check_reproduction.py artifacts/submission.csv artifacts/reproduced/submission.csv
.\.venv\Scripts\python.exe scripts/verify_delivery.py
```

Verify_delivery проверяет сохранённые результаты и не обучает модель.
Ему нужны исходные baseline/control_original, контрольный submission, manifest и существующие snapshots/cache.
Недостающие файлы приводят к ошибке. Доступны --reproduced-dir и --report-dir для отдельных каталогов результата и отчёта.
Допуск сравнения: rtol=1e-12, atol=1e-9.

## Выполненные проверки

На Windows / Python 3.12.10 проверены новая отдельная .venv, установка, pip check, train и predict.
До добавления модели: 52 passed, 4 skipped. После обучения: 56 passed.
Все 597 прогнозов совпали с контрольными, максимальное отличие 3,64e-12.
Проверены реальный расчёт Streamlit, upload CSV и оба экспорта.

Опубликованный исходный состав дополнительно проверялся через настоящий GitHub-клон с локальными CSV
и доверенной сохранённой моделью; повторного обучения в этом клоне не было.
После изменения состава публикации проверку нового удалённого состояния следует повторить.
Native Linux текущего состояния и реальный Kaggle score не подтверждены.
