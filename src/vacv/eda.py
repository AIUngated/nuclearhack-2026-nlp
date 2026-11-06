from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import make_features, max_train_similarity, molecular_metadata
from .data import read_table, validate_frame
from .utils import environment, sha256_file, write_json


def dataset_summary(frame, metadata):
    return {"rows": len(frame), "columns": frame.columns.tolist(),
            "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing": {k: int(v) for k, v in frame.isna().sum().items()},
            "raw_smiles_unique": int(frame["SMILES"].nunique()),
            "canonical_unique": int(metadata["canonical_smiles"].nunique()),
            "compound_groups": int(metadata["compound_group"].nunique()),
            "scaffolds": int(metadata["scaffold"].nunique()),
            "acyclic_rows": int(metadata["scaffold"].eq("__ACYCLIC__").sum())}


def eda_pipeline(train_path, test_path, config, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test = read_table(train_path), read_table(test_path)
    validate_frame(train, config, training=True)
    validate_frame(test, config)
    x_train, train_meta = make_features(train[config["smiles_column"]].tolist(), config)
    x_test, test_meta = make_features(test[config["smiles_column"]].tolist(), config)
    y = train[config["target_column"]].to_numpy(dtype=float)
    combined = train_meta.copy()
    combined["row_id"], combined["target"] = np.arange(len(train)), y
    conflicts = combined.groupby("compound_group", as_index=False).agg(
        rows=("target", "size"), target_min=("target", "min"), target_max=("target", "max"))
    conflicts = conflicts[conflicts["rows"] > 1].copy()
    conflicts["spread"] = conflicts["target_max"] - conflicts["target_min"]
    conflicts.to_csv(output / "duplicate_groups.csv", index=False)
    train_meta.assign(row_id=np.arange(len(train))).to_csv(output / "train_molecules.csv", index=False)
    test_meta.assign(row_id=np.arange(len(test))).to_csv(output / "test_molecules.csv", index=False)
    similarity = max_train_similarity(x_test, x_train, config["n_bits"])
    overlap = {}
    for key in ("canonical_smiles", "compound_group", "scaffold"):
        reference = set(train_meta[key])
        overlap[key] = {"unique_overlap": len(reference & set(test_meta[key])),
                        "test_rows_seen": int(test_meta[key].isin(reference).sum())}
    sentinel = y == 10000
    label_review = {"rows_equal_10000": int(sentinel.sum()), "zero_labels": int((y == 0).sum())}
    if "IC50_uM_VACV" in train:
        missing_ic50 = train["IC50_uM_VACV"].isna().to_numpy()
        label_review.update(missing_ic50_rows=int(missing_ic50.sum()),
                            missing_ic50_exactly_matches_10000=bool(np.array_equal(missing_ic50, sentinel)))
    # This check is strictly a TRAIN-LABEL AUDIT. Never reconstruct test SI.
    if {"CC50_uM", "IC50_uM_VACV"} <= set(train.columns):
        cc = pd.to_numeric(train["CC50_uM"], errors="coerce").to_numpy()
        ic = pd.to_numeric(train["IC50_uM_VACV"], errors="coerce").to_numpy()
        available = np.isfinite(cc) & np.isfinite(ic) & (cc > 0) & (ic > 0)
        audit = pd.DataFrame({"row_id": np.flatnonzero(available), "provided_si": y[available],
                              "train_only_cc50_over_ic50": cc[available] / ic[available]})
        audit["matches_with_1pct_tolerance"] = np.isclose(audit["provided_si"], audit["train_only_cc50_over_ic50"], rtol=.01, atol=.01)
        audit.to_csv(output / "train_label_ratio_audit.csv", index=False)
        label_review["ratio_comparable_rows"] = len(audit)
        label_review["ratio_mismatches_1pct"] = int((~audit["matches_with_1pct_tolerance"]).sum())
    summary = {"train": dataset_summary(train, train_meta), "test": dataset_summary(test, test_meta),
               "target": {"mean": float(y.mean()), "std": float(y.std(ddof=1)),
                          "quantiles": {str(q): float(np.quantile(y, q)) for q in (0, .25, .5, .75, .9, .99, 1)}},
               "label_review": label_review, "train_test_overlap": overlap,
               "test_max_train_tanimoto_quantiles": {str(q): float(np.quantile(similarity, q)) for q in (0, .25, .5, .75, 1)},
               "train_sha256": sha256_file(train_path), "test_sha256": sha256_file(test_path),
               "environment": environment(), "invalid_smiles": {"train": 0, "test": 0}}
    write_json(output / "summary.json", summary)
    # Exact data supplied by the competition, exported in the requested collection schema.
    training_table = pd.DataFrame({"smiles": train[config["smiles_column"]],
                                   "IC50": train.get("IC50_uM_VACV"), "CC50": train.get("CC50_uM"),
                                   "SI": y, "doi": "", "concentration_unit": "uM",
                                   "source": "organizer_train.csv", "source_row": np.arange(len(train)),
                                   "si_definition": "organizer_provided; censoring_not_documented"})
    training_table.to_csv(output / "training_dataset.csv", index=False)
    text = f"""# EDA: Nuclear IT Hack 2026 / VACV

## Требования и входные файлы

- Train: **{len(train)}** строк; test: **{len(test)}** строк.
- Цель: `SI VACV`, регрессия; основная метрика: **MSE в исходной шкале SI**.
- Признаки: только `SMILES`. `CC50_uM`, `IC50_uM_VACV`, `Title`, `strain`, `Culture` не передаются модели.
- Образец submission задаёт только имена колонок и порядок `index`. Его числовые значения не являются метками.
- Все SMILES обработаны установленной версией RDKit; некорректных: 0.

## Химические дубликаты и утечки

| Показатель | Train | Test |
|---|---:|---:|
| Строки | {len(train)} | {len(test)} |
| Уникальные исходные SMILES | {train[config['smiles_column']].nunique()} | {test[config['smiles_column']].nunique()} |
| Канонические структуры с учётом стереохимии | {train_meta.canonical_smiles.nunique()} | {test_meta.canonical_smiles.nunique()} |
| Родительские структуры без стереохимии | {train_meta.compound_group.nunique()} | {test_meta.compound_group.nunique()} |
| Murcko scaffolds | {train_meta.scaffold.nunique()} | {test_meta.scaffold.nunique()} |

Между train/test совпадают {overlap['canonical_smiles']['unique_overlap']} канонических структур и
{overlap['compound_group']['unique_overlap']} родительских групп. Это обнаруженное свойство предоставленного разбиения.
Метки test не известны и не восстанавливаются из запрещённых колонок.

Валидация `compound`: одинаковые родительские структуры, соли и стереоварианты остаются в одном fold.
Валидация `scaffold`: весь Murcko scaffold остаётся в одном fold. Все ациклические структуры при их наличии образуют одну группу.
Обычный random split по строкам не используется. Разбиение по публикациям пока невозможно: DOI в исходных файлах отсутствует.
Группы с повторными измерениями и расхождением меток сохранены в `duplicate_groups.csv`; конфликтные метки не усредняются до CV.

## Распределение и качество меток

- Среднее SI: {y.mean():.3f}; медиана: {np.median(y):.3f}; максимум: {y.max():.3f}.
- SI ровно 10 000: **{sentinel.sum()} / {len(y)}**; SI = 0: **{(y == 0).sum()}**.
- Проверка исходных концентраций выполняется только для аудита train: `train_label_ratio_audit.csv`.
- Сводка проверки: `{label_review}`.

Совпадение SI = 10 000 с отсутствием IC50 требует объяснения организаторов. Возможны особенности
кодирования/цензурирования, но это **не установленный факт**. Метки сохраняются без исправления.
Отсутствие IC50 не используется ни как признак, ни как правило для предсказания.
Сравнение raw/log-моделей проводится по raw MSE; логарифмирование не меняет официальную метрику.

## Область применимости

Медиана максимального Tanimoto test→train: {np.median(similarity):.3f}.
Сходство вычисляется по Morgan radius={config['radius']}, {config['n_bits']} bits.
Это диагностический показатель химического сходства, а не вероятность правильного ответа.
Данные включают разные клеточные культуры и штаммы. Одна структура может иметь разные SI в разных условиях;
SMILES-only модель не может полностью объяснить такую вариабельность.
Цитотоксичность не равна клинической безопасности; отдельная модель безопасности здесь не обучена.

## Ресурсы и воспроизводимость

CPU-only pipeline; все модели имеют фиксированный seed, `n_jobs={config['n_jobs']}`.
SHA256 исходных файлов и версии библиотек: `summary.json`. Окружение не предоставляет подтверждения доступной GPU.
Собранная таблица фактических обучающих данных: `training_dataset.csv` (DOI оставлены пустыми, а не придуманы).
График: `target_distribution.png`, если установлен matplotlib.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].hist(y, bins=35, color="#176b87")
        axes[0].set(xlabel="Official SI", ylabel="Training rows", title="Raw target: MSE is evaluated here")
        axes[1].hist(np.log1p(y), bins=35, color="#d27645")
        axes[1].set(xlabel="log(1 + SI)", ylabel="Training rows", title="Same labels, logarithmic view")
        fig.savefig(output / "target_distribution.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("Optional matplotlib is absent; report and tables were produced without plots.")
    print(f"EDA saved to {output}", flush=True)
    return summary
