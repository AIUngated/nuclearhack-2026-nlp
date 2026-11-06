#!/usr/bin/env python3
"""Build report tables directly from saved experimental results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd


def markdown_table(frame, numeric_digits=3):
    labels = frame.columns.tolist()
    rows = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if pd.isna(value):
                formatted.append("—")
            elif isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.{numeric_digits}f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def main():
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    baseline = pd.read_csv(ROOT / "artifacts/baseline/metrics.csv")
    runs = [ROOT / "artifacts/baseline", ROOT / "artifacts/stability_seed7", ROOT / "artifacts/stability_seed2026"]
    frames = [pd.read_csv(run / "metrics.csv") for run in runs if (run / "metrics.csv").exists()]
    repeated = pd.concat(frames, ignore_index=True)
    repeated.to_csv(output / "all_experiments.csv", index=False)
    repeated = repeated[repeated.model.isin(["dummy_mean", "random_forest", "tanimoto_1"])]
    stability = repeated.groupby(["split", "model"], as_index=False).agg(
        seeds=("seed", "nunique"), mse_mean=("mse", "mean"), mse_std=("mse", "std"), rmse_mean=("rmse", "mean"))
    stability.to_csv(output / "seed_stability.csv", index=False)
    wide = baseline.pivot(index="model", columns="split", values="mse").reset_index().sort_values("compound")
    oof = pd.read_csv(ROOT / "artifacts/baseline/oof_predictions.csv")
    chosen = oof[oof.model.eq("random_forest") & oof.split.eq("compound")].copy()
    chosen["squared_error"] = (chosen.target - chosen.prediction) ** 2
    chosen["absolute_error"] = (chosen.target - chosen.prediction).abs()
    chosen["similarity_bin"] = pd.cut(chosen.max_train_tanimoto, [0, .3, .5, .7, .85, 1.000001], include_lowest=True)
    bins = chosen.groupby("similarity_bin", observed=True).agg(rows=("row_id", "size"), mse=("squared_error", "mean"), mae=("absolute_error", "mean"))
    bins.to_csv(output / "error_by_similarity.csv")
    chosen["label_cohort"] = np.where(chosen.target.eq(10000), "official_SI_equal_10000", "other_official_labels")
    chosen.groupby("label_cohort").agg(rows=("row_id", "size"), mse=("squared_error", "mean"), mae=("absolute_error", "mean")).to_csv(output / "error_by_label_cohort.csv")
    worst = chosen.sort_values("squared_error", ascending=False).head(20)
    worst.to_csv(output / "worst_oof_errors.csv", index=False)
    contribution = float(chosen.loc[chosen.target.idxmax(), "squared_error"] / chosen.squared_error.sum())
    dummy = float(baseline.loc[baseline.model.eq("dummy_mean") & baseline.split.eq("compound"), "mse"].iloc[0])
    rf = float(baseline.loc[baseline.model.eq("random_forest") & baseline.split.eq("compound"), "mse"].iloc[0])
    report = f"""# Реально выполненные эксперименты

## Протокол

Train: 320 строк, только SMILES-признаки. Morgan radius=2, 2048 bits, chirality=True + 16 дескрипторов.
Пять folds: `compound` (parent без стереохимии) и `scaffold` (Murcko).
Ни test-метки, ни концентрации/пропуски/метаданные, ни значения sample-submit не использовались.
Все raw/log варианты сравниваются по MSE в исходной шкале SI; отрицательные прогнозы ограничиваются нулём.
Основной split для выбора заранее указан в configs/baseline.json: compound.
Препроцессинг обучается внутри folds. Подробные folds, метрики и OOF находятся в artifacts/baseline/.

## Сравнение 10 моделей, seed=42

{markdown_table(wide)}

Random Forest выбран для поставляемого baseline: MSE **{rf:,.3f}** против **{dummy:,.3f}**
у среднего, относительное уменьшение MSE **{100 * (1 - rf / dummy):.2f}%** на compound CV.
На scaffold CV у RF MSE выше, чем у Tanimoto Kernel Ridge: обобщение на новые химические каркасы труднее.

## Проверка seeds 42, 7, 2026

{markdown_table(stability)}

Стандартное отклонение между seeds показывает чувствительность к разбиению. Это не доверительный интервал:
строки и folds зависимы, набор мал, часть кандидатов уже выбрана по той же CV.
Полная таблица отдельных запусков: all_experiments.csv. Основной submission использует RF seed=42,
а не выбор лучшего seed по его результату. Повторные запуски служат проверкой устойчивости.

## Анализ ошибок

Одна строка с максимальным SI объясняет **{100 * contribution:.2f}%** суммарной squared error
RF compound OOF. Удалять этот объект ради улучшения числа нельзя без доказанной ошибки источника.
Ошибки по диапазонам Tanimoto: error_by_similarity.csv. Группы SI=10000 и остальных меток:
error_by_label_cohort.csv. Самые большие промахи: worst_oof_errors.csv.
Выделение групп по метке используется только для анализа ошибок, а не как дополнительный признак.

## Чего результаты не доказывают

- Это локальная CV, а не Kaggle score. Отправка в Kaggle в этом чате не выполнялась.
- Выбор модели и оценка использовали одну CV; итоговое число оптимистично относительно полностью независимого теста.
- Ни один внешний ChEMBL label не добавлялся в baseline: не подтверждена совместимость определений/условий.
- Нет калиброванных интервалов и независимой модели токсичности.
- Высокий прогноз SI не доказывает, что соединение активно: отношение может быть высоким при разных абсолютных концентрациях.
- SI в отдельных культурах не переносится автоматически на человека.

Следующий эксперимент: проверить строго курированные внешние данные с fold-specific исключением
пересечений. Затем проверить заранее фиксированный ансамбль RF + Tanimoto на тех же folds и нескольких seeds.
"""
    (output / "experiment_report.md").write_text(report, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for axis, mode in zip(axes, ["compound", "scaffold"]):
            part = baseline[baseline.split.eq(mode)].sort_values("mse", ascending=False)
            colors = ["#d27645" if name == "dummy_mean" else "#176b87" for name in part.model]
            axis.barh(part.model, part.mse / 1e6, color=colors)
            axis.set(xlabel="OOF MSE, millions (lower is better)", title=f"{mode.capitalize()} 5-fold CV · seed 42")
            axis.grid(axis="x", alpha=.2)
        fig.savefig(output / "model_comparison.png", dpi=170)
        plt.close(fig)
    except ImportError:
        pass
    print(f"Experiment report saved to {output}")


if __name__ == "__main__":
    main()
