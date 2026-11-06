# Проверенные требования

## Приоритет источников

1. Предоставленный документ `docs/sources/problem_statement.docx`: регрессия SI,
   обязательный скрипт открытых данных, обучающая таблица, GitHub и воспроизведение submission.
2. Требования вкладок Kaggle, предоставленные командой 6 ноября 2026 года:
   MSE, `index,SI VACV`, запрет использовать `CC50_uM` и `IC50_uM_VACV` при моделировании.
3. Реальные CSV в `data/raw`: фактическая схема и порядок строк.
4. Публичная страница https://nuclearhack.mephi.ru/ : кейс и расписание.

Приглашение в приватное соревнование хранится у команды и не публикуется в репозитории.
Текущую редакцию правил следует сверять в приватном соревновании Kaggle.

## Что сдавать

| Объект | Подтверждённое требование | Реализация в пакете |
|---|---|---|
| Предсказания | CSV `index,SI VACV` | `artifacts/submission.csv`, 597 строк |
| Модель | SMILES → SI VACV | Morgan + descriptors, Random Forest baseline |
| Метрика | MSE | raw SI; MAE/RMSE/R² — дополнительные |
| Скрипт обучения/проверки | обязателен | `scripts/run.py train`, CV в `src/vacv/evaluation.py` |
| Воспроизводимый pipeline | обязателен | `scripts/run.py reproduce`, pinned requirements, SHA256 |
| Выгрузка и валидация открытых данных | обязательна | `src/vacv/collect.py`, CLI collect, HTTP snapshots |
| Собранная таблица | SMILES, IC50, CC50, SI, DOI при наличии | `external_candidates.csv`; поля намеренно разрежены, если измерения не парные |
| Проверенные внешние пары | совместимые endpoint и контекст | `external_curated.csv`: 5 пар с DOI, пересечения отмечены |
| Фактические обучающие данные | обязательны | `reports/eda/training_dataset.csv`; DOI не выдуманы |
| Исходный код | ссылка GitHub | https://github.com/AIUngated/nuclearhack-2026-nlp |
| Допуск на защиту | до пяти команд, рейтинг и воспроизводимость | итог зависит от Kaggle и организаторов |

## Ограничения и неопределённости

- SI = CC50/IC50 — определение из задания. Значения и единицы уже предоставлены в train;
  в `train_label_ratio_audit.csv` проверены только доступные train-измерения.
- В train 81 значение SI=10000; все они совпадают с отсутствием IC50.
  Нельзя заключить, что это реальная точная биологическая селективность.
- Для 237 строк с положительными CC50/IC50 метки согласуются с отношением в пределах 1% / 0.01.
- Нулевые SI: 2 строки. Не заменять их автоматически без объяснения источника.
- Клеточные культуры неоднородны, включая не человеческие линии. SI не означает безопасность для человека.
- `sample_submit.csv` содержит произвольные числа: они не используются для подгонки/оценки.
- Наличие концентраций в test не разрешает их использование: запрет организаторов распространяется и на их пропуски.
- Разрешение внешних предобученных моделей, правила загрузки данных и лимит Kaggle submissions отдельно не предоставлены.
- Финальное расписание берётся у организаторов. Рабочий дедлайн команды: 7 ноября 2026, 21:00 по Москве;
  защита 8 ноября, 11:00 по Москве.

## Научные источники реализации

- ChEMBL API: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
- ChEMBL assay/activity semantics: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions
- RDKit: https://www.rdkit.org/docs/GettingStartedInPython.html
- Group split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

Данные ChEMBL получены реально; запросы, время получения, DOI и SHA256 лежат в `collection_manifest.json`.
Все предсказания этого проекта — вычислительные гипотезы, не экспериментальная проверка лекарства.
