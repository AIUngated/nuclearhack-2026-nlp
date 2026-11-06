# NLP · VACV Selectivity Prediction

Проект команды NLP для Nuclear IT Hack 2026, НИЯУ МИФИ: прогноз **SI VACV по SMILES**.
Задача регрессии, официальная метрика — **MSE в исходной шкале SI**.

## Метод

RDKit: FragmentParent → Uncharger → Morgan radius=2, 2048 bits с учётом стереохимии и 16 дескрипторов.
Основная модель — Random Forest, 300 деревьев, seed=42, CPU, n_jobs=2.

В признаки поступает только SMILES. CC50_uM, IC50_uM_VACV, их пропуски и отношение,
Title, Culture, strain, index и числа из sample-submit исключены.
Index используется для выравнивания выходного CSV.

## Данные

Train содержит 320 строк, test — 597. После стандартизации train содержит 311 parent-групп
и 165 Murcko scaffolds. Данные приватного соревнования, модели и контрольные артефакты
распространяются отдельно от исходников.

Получите разрешённые train.csv, test.csv и sample_submit.csv и поместите их в data/raw/.
Метки SI=10000 (81 строка) и нулевые метки (2 строки) сохраняются без изменений до разъяснения организаторов.

## Установка и запуск

Рекомендован Python 3.12. Следующие команды предназначены для новой папки с кодом.

Windows PowerShell:

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

Linux/macOS используют те же Python-команды после создания окружения и активации
`source .venv/bin/activate`. Финальный набор проверен на Windows / Python 3.12.10;
собственный native Linux-прогон текущей версии не выполнялся.
Внешнее воспроизведение коммита 4f48d61 на Linux, переданное владельцем проекта,
зафиксировано отдельно: [внешний аудит](reports/external_audits/linux_4f48d61.json).

Train сохраняет artifacts/baseline/model.joblib; predict создаёт artifacts/submission.csv.
В существующей рабочей папке используйте отдельный output для повторного обучения.

## Воспроизводимость и валидация

Пять compound folds удерживают parent-структуры, соли и стереоварианты в одной группе.
Отдельные scaffold folds удерживают каждый Murcko scaffold целиком.
Все обучаемые преобразования fit выполняются внутри тренировочной части fold.

Для рабочей папки с доверенным контрольным пакетом:

```powershell
.\.venv\Scripts\python.exe scripts/run.py reproduce --output artifacts/reproduced
.\.venv\Scripts\python.exe scripts/check_reproduction.py artifacts/submission.csv artifacts/reproduced/submission.csv
.\.venv\Scripts\python.exe scripts/verify_delivery.py
```

Проверяющий скрипт сверяет конфигурацию, OOF/folds, обе сохранённые модели и фактическую модель демо,
вход только по SMILES, schema CSV, порядок index и контрольные суммы.
Для него нужны доверенные контрольные артефакты и сохранённые snapshots; один Git-клон их не содержит.
Пустые CSV, NaN/Inf, отрицательные прогнозы и повторные index отклоняются.

| Локальная CV RF, seed=42 | Raw MSE |
|---|---:|
| compound, 5 folds | 14 573 172,262681 |
| scaffold, 5 folds | 20 682 986,252401 |

На исходном scaffold split Tanimoto Kernel Ridge: MSE 19 055 199,440157.
Модель выбиралась по compound CV. Выбор и оценка используют одну CV, поэтому это не независимая оценка.
Реальный Kaggle score пока не зафиксирован. Сводка проверок: [reports/verification_summary.json](reports/verification_summary.json).

У четырёх дополнительных кандидатов наблюдаются численные различия между платформами.
Все отличия фиксируются в таблице сравнения; выбранный RF воспроизводится с rtol=1e-12, atol=1e-9.
Флаг --strict-all-candidates дополнительно требует совпадения всех кандидатов.

## Streamlit

Один SMILES или CSV → сохранённая модель → прогноз, структура, сортировка, распределение и экспорт.
Полный экспорт сохраняет исходный порядок строк. Для полного официального test доступен отдельный
CSV ровно с колонками index,SI VACV в порядке sample_submit.csv.

Tanimoto описывает химическое сходство с train, а не вероятность или доверительный интервал.
Прогноз SI не подтверждает эффективность или клиническую безопасность.

## Внешние источники

Сборщик ChEMBL и HTTP snapshots содержат 6148 измерений, 135 публикаций и 532 assay.
Это не 6148 меток SI. CC50/EC50 не подменяется CC50/IC50, цензурированные границы не считаются точными значениями.
Пять сопоставленных пар из одной статьи пересекаются с train/test и не добавлены в обучение.
Их независимое ручное подтверждение остаётся незавершённым.

Базовая модель обучена только на исходном train. Новое расширение данных и ансамбль отложены.
Существующий сборщик доступен через `python scripts/run.py collect`; при наличии кэша — `collect --offline`.

## Документация

- [Требования и ограничения](docs/requirements.md)
- [Запуск из нового клона](docs/clean_clone.md)
- [Проверка на Windows](docs/windows_runbook.md)
- [Статус курации](docs/curation_note.md)
- [План проекта](docs/implementation_plan.md)
- [Структура защиты](presentation/defense_outline.md)
- [Локальный комплект для сдачи](docs/final_submission.md)
- [Краткий запуск демо](docs/demo_quickstart.md)

Публикуемый состав исходников задан в publication_files.txt. Исходные данные, модели и контрольный
пакет сохраняются отдельно; секреты и приватные документы в репозиторий не входят.
