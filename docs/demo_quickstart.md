# Быстрый запуск демо

В локальном комплекте модель уже находится в `artifacts/baseline/model.joblib`.
Для демонстрации повторное обучение не требуется. Все команды выполняются из корня
распакованной папки `vacv_submission`, где находятся `pyproject.toml` и `app/`.

## Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 6 -r requirements-app.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

Откройте http://127.0.0.1:8501. Введите SMILES или загрузите CSV с колонкой `SMILES`,
затем нажмите «Рассчитать прогноз». Если ввод пуст, используется вложенный `data/raw/test.csv`.

«Скачать все прогнозы CSV» сохраняет порядок входных строк. Для полного test с официальными
index доступна кнопка «Скачать submission для Kaggle»: только `index,SI VACV` в порядке шаблона.
Сортировка и фильтр на экране не меняют полный экспорт.

## Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m pip install -e . --no-deps
python -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

Собственные проверки установки и UI выполнены на Windows. Внешний Linux-аудит относится
к обучению, прогнозам и тестам опубликованного коммита; отдельный браузерный прогон на Linux
в полученном сообщении не описан.

Tanimoto показывает структурное сходство с train, а не калиброванную уверенность.
Прогноз SI не является подтверждением эффективности или клинической безопасности.
