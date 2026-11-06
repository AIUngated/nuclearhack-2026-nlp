from pathlib import Path
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from vacv.data import submission_from_template
from vacv.pipeline import load_bundle, predict_frame

st.set_page_config(page_title="NLP · VACV Selectivity", page_icon="🧬", layout="wide")
st.title("NLP · Оценка селективности по структуре молекулы")
st.caption("Nuclear IT Hack 2026 · VACV · SMILES → прогноз SI")
st.info("Прогноз SI отражает обучающие данные и условия измерений. Он не доказывает эффективность препарата или клиническую безопасность.")

model_path = ROOT / "artifacts/baseline/model.joblib"
if not model_path.exists():
    st.error("Сначала обучите модель: python scripts/run.py train")
    st.stop()


@st.cache_resource
def model_bundle(path, modified_time):
    return load_bundle(path)


try:
    model_version = model_path.stat().st_mtime_ns
    bundle = model_bundle(str(model_path), model_version)
except (OSError, ValueError) as error:
    st.error(f"Не удалось загрузить сохранённую модель: {error}")
    st.stop()
st.sidebar.write(f"Модель: **{bundle['model_name']}**")
st.sidebar.write("Вход модели: только SMILES")
st.sidebar.caption("Близость по Tanimoto — диагностика области применимости, не вероятность и не доверительный интервал.")
uploaded = st.file_uploader("CSV с колонкой SMILES", type=["csv"])
single = st.text_input("Или введите один SMILES", placeholder="Например: c1ccccc1")

try:
    if single.strip():
        frame = pd.DataFrame({"SMILES": [single.strip()]})
    elif uploaded is not None:
        frame = pd.read_csv(uploaded)
    else:
        demo_path = ROOT / "data/raw/test.csv"
        if demo_path.exists():
            st.caption("Для расчёта выбран предоставленный test.csv. Можно загрузить свой файл или ввести SMILES.")
            frame = pd.read_csv(demo_path)
        else:
            st.caption("Для примера выбран SMILES c1ccccc1. Можно загрузить свой CSV или ввести другую структуру.")
            frame = pd.DataFrame({"SMILES": ["c1ccccc1"]})
    if "SMILES" not in frame:
        raise ValueError("CSV должен содержать колонку SMILES.")
    if frame.empty:
        raise ValueError("CSV не содержит строк для расчёта.")
    signature = hashlib.sha256(
        f"{model_path}:{model_version}:".encode() + frame.to_csv(index=False).encode()).hexdigest()
    if st.session_state.get("result_source") != signature:
        st.session_state.pop("result", None)
    if st.button("Рассчитать прогноз", type="primary"):
        with st.spinner("Обрабатываем структуры и запускаем модель"):
            predictions, diagnostics = predict_frame(bundle, frame)
        result = pd.DataFrame({"SMILES": frame["SMILES"].to_numpy(), "prediction_si": predictions,
                               "max_train_tanimoto": diagnostics["max_train_tanimoto"].to_numpy(),
                               "seen_structure": diagnostics["seen_canonical_structure"].to_numpy()})
        if "index" in frame:
            result.insert(0, "index", frame["index"].to_numpy())
        st.session_state["result"] = result
        st.session_state["result_source"] = signature
    if "result" in st.session_state:
        result = st.session_state["result"]
        left, middle, right = st.columns(3)
        left.metric("Соединений", len(result))
        middle.metric("Медиана прогноза SI", f"{result.prediction_si.median():.2f}")
        right.metric("Медиана ближайшего Tanimoto", f"{result.max_train_tanimoto.median():.3f}")
        min_similarity = st.slider("Минимальная близость к train", 0.0, 1.0, 0.0, 0.05)
        sort_columns = {"По прогнозу SI": "prediction_si", "По близости к train": "max_train_tanimoto"}
        order = sort_columns[st.selectbox("Сортировка", list(sort_columns))]
        shown = result[result.max_train_tanimoto >= min_similarity].sort_values(order, ascending=False)
        st.dataframe(shown, hide_index=True, width="stretch", column_config={
            "prediction_si": st.column_config.NumberColumn("Прогноз SI", format="%.3f"),
            "max_train_tanimoto": st.column_config.NumberColumn("Ближайший Tanimoto", format="%.3f"),
            "seen_structure": "Структура встречалась в train",
        })
        if shown.empty:
            st.info("Нет соединений с выбранной минимальной близостью к train.")
        counts = result.prediction_si.value_counts(bins=20, sort=False)
        histogram = pd.DataFrame({"Прогноз SI": [interval.mid for interval in counts.index],
                                  "Соединений": counts.to_numpy()})
        st.bar_chart(histogram, x="Прогноз SI", y="Соединений")
        st.download_button("Скачать все прогнозы CSV", result.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        template_path = ROOT / "data/raw/sample_submit.csv"
        if "index" in result and template_path.exists():
            template = pd.read_csv(template_path)
            if (len(result) == len(template) and result["index"].is_unique
                    and set(result["index"]) == set(template["index"])):
                submission = submission_from_template(
                    template, result, result.prediction_si.to_numpy(), bundle["config"])
                st.download_button("Скачать submission для Kaggle",
                                   submission.to_csv(index=False).encode("utf-8"),
                                   "submission.csv", "text/csv")
        if len(shown):
            selected = st.selectbox("Структура молекулы", shown.index.tolist())
            mol = Chem.MolFromSmiles(result.loc[selected, "SMILES"])
            if mol is not None:
                st.image(Draw.MolToImage(mol, size=(600, 350)))
except (ValueError, KeyError, RuntimeError) as error:
    st.session_state.pop("result", None)
    st.session_state.pop("result_source", None)
    st.error(str(error))
