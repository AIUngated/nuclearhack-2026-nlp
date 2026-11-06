# Установка и проверка на Windows

Проект использует CPU и Python 3.12. Проверенное окружение: Windows 11 / Python 3.12.10.
Исходные версии numpy, pandas, scipy, scikit-learn, joblib и RDKit закреплены в requirements.txt.

## Окружение

Из корня проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 6 -r requirements-app.txt -r requirements-reports.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
```

Активация окружения и изменение ExecutionPolicy не требуются.
Для повторения полного установленного набора зависимостей есть reports/requirements-windows-lock.txt.

## Новая папка с кодом

После помещения разрешённых CSV в data/raw/:

```powershell
.\.venv\Scripts\python.exe scripts/run.py train
.\.venv\Scripts\python.exe scripts/run.py predict
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

Демо загружает artifacts/baseline/model.joblib.
При наличии test.csv можно рассчитать все 597 строк; без него доступен один пример SMILES.
Общий экспорт сохраняет порядок входа, submission для Kaggle выравнивается по официальному шаблону.

## Существующая папка с контрольным пакетом

Исходный baseline и submission сохраняются отдельно.
Повторное обучение записывается в artifacts/reproduced/:

```powershell
.\.venv\Scripts\python.exe scripts/run.py reproduce --output artifacts/reproduced
.\.venv\Scripts\python.exe scripts/check_reproduction.py artifacts/submission.csv artifacts/reproduced/submission.csv
.\.venv\Scripts\python.exe scripts/verify_delivery.py
```

Проверка включает выбранный RF, OOF/folds, загрузку моделей, фактическую модель демо,
CSV schema/index, SMILES-only вход и существующие snapshots.
Для отдельного отчёта используйте --report-dir; для другого результата обучения — --reproduced-dir.

Все 597 прогнозов прошли допуск rtol=1e-12, atol=1e-9.
Compound MSE RF: 14 573 172,262681; scaffold MSE: 20 682 986,252401.
Это локальная CV. Итоговые проверки перечислены в reports/verification_summary.json.

У Ridge, kNN и Extra Trees log сохраняются отличия между исходным Linux и Windows.
Их причины полностью не установлены; для изменённых kNN-прогнозов обнаружены равные сходства
на границе выбора соседей. Код моделей и официальные SI-метки сохраняются.
