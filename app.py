"""Local Streamlit interface for ComplaintCompass."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from complaint_compass.config import (
    ARTIFACTS_DIR,
    ENSEMBLE_MODEL_NAME,
    MODEL_DESCRIPTIONS,
    PRODUCT_LABELS,
    REPORTS_DIR,
)
from complaint_compass.inference import ArtifactError, ComplaintPredictor
from complaint_compass.text import TextValidationError

st.set_page_config(
    page_title="ComplaintCompass",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_predictor(artifacts_path: str) -> ComplaintPredictor:
    return ComplaintPredictor(Path(artifacts_path))


def _metric_frame(payload: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in payload.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "Macro precision": metrics.get("macro_precision"),
                "Macro recall": metrics.get("macro_recall"),
                "Macro F1": metrics.get("macro_f1"),
                "Weighted F1": metrics.get("weighted_f1"),
                "Size (MB)": (
                    metrics.get("artifact_size_bytes", 0) / 1_048_576
                ),
                "Latency (ms/text)": metrics.get("mean_inference_latency_ms"),
            }
        )
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False)


def _load_metrics() -> tuple[str, dict[str, dict[str, Any]]] | None:
    candidates = (
        ("Sealed test results", REPORTS_DIR / "test_metrics.json"),
        ("Validation results", REPORTS_DIR / "validation_metrics.json"),
    )
    for label, path in candidates:
        if path.exists():
            return label, json.loads(path.read_text(encoding="utf-8"))
    return None


def _classify_tab(predictor: ComplaintPredictor) -> None:
    st.subheader("Classify a complaint")
    st.write(
        "Paste an English consumer-finance complaint. Text is processed in memory "
        "and is not stored or logged by this application."
    )
    model_name = st.selectbox(
        "Model",
        options=list(predictor.available_models),
        index=list(predictor.available_models).index(predictor.default_model),
        format_func=lambda name: f"{name} — {MODEL_DESCRIPTIONS.get(name, name)}",
    )
    if model_name == ENSEMBLE_MODEL_NAME:
        metadata = predictor.registry.get("models", {}).get(model_name, {})
        weights = metadata.get("weights", {})
        st.info(
            "This model uses validation-selected fixed soft-voting weights and "
            "currently has the highest validation macro-F1."
        )
        with st.expander("How the weighted ensemble combines models"):
            rows = [
                {
                    "Model": name,
                    "Role": MODEL_DESCRIPTIONS.get(name, name),
                    "Weight": f"{float(weights[name]):.0%}",
                }
                for name in metadata.get("base_models", [])
                if name in weights
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif model_name == "adaptive_fusion":
        st.caption(
            "ARUF adapts each member's influence using category-level validation "
            "reliability, prediction uncertainty, and agreement."
        )
    narrative = st.text_area(
        "Complaint narrative",
        height=220,
        max_chars=2_000,
        placeholder=(
            "Example: I disputed an account on my credit report, but the incorrect "
            "information is still showing after the investigation..."
        ),
    )
    if st.button("Classify complaint", type="primary", width="stretch"):
        try:
            result = predictor.predict(narrative, model_name=model_name)
        except TextValidationError as exc:
            st.warning(str(exc))
            return
        except ArtifactError as exc:
            st.error(str(exc))
            return

        st.success(result["label"])
        st.metric("Model confidence", f"{result['confidence']:.4%}")
        chart = pd.DataFrame(result["top_categories"]).set_index("label")
        st.caption("Top three candidate categories")
        st.bar_chart(chart["probability"], horizontal=True)
        st.info(
            "This confidence is a model estimate, not a guarantee. A human should "
            "review uncertain or consequential routing decisions."
        )


def _comparison_tab() -> None:
    st.subheader("Compare models")
    loaded = _load_metrics()
    if loaded is None:
        st.info(
            "No evaluation report is available. Run "
            "`python -m complaint_compass.evaluate` after training."
        )
        return
    label, metrics = loaded
    st.caption(label)
    st.dataframe(
        _metric_frame(metrics),
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Macro precision": st.column_config.NumberColumn(format="%.4f"),
            "Macro recall": st.column_config.NumberColumn(format="%.4f"),
            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
            "Weighted F1": st.column_config.NumberColumn(format="%.4f"),
            "Size (MB)": st.column_config.NumberColumn(format="%.1f"),
            "Latency (ms/text)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    for model_name in metrics:
        image_path = REPORTS_DIR / f"confusion_matrix_{model_name}.png"
        if image_path.exists():
            with st.expander(f"Confusion matrix — {model_name}"):
                st.image(str(image_path), width="stretch")


def _about_tab() -> None:
    st.subheader("About the data and intended use")
    st.write(
        "ComplaintCompass is an academic routing prototype trained on public "
        "Consumer Financial Protection Bureau complaint narratives. It predicts "
        "one of these six product categories:"
    )
    for label in PRODUCT_LABELS:
        st.markdown(f"- {label}")
    st.markdown(
        """
        **Limitations**

        - Narratives are published only when consumers opt in and after CFPB scrubbing.
        - The CFPB does not verify the narratives, and the data is not a statistical
          sample of all consumer experiences.
        - The data concerns the United States and may not generalize to other regions.
        - Inputs are limited to English and 2,000 characters.
        - Predictions must not be used to judge consumers, companies, or complaint merit.
        - The combination-method test results are exploratory because the original
          holdout had already been inspected before the broader analysis was completed.
        - Adaptive fusion is a project-specific combination of established ensemble and
          uncertainty concepts, not a claim of global algorithmic novelty.

        [Official CFPB database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
        """
    )


def main() -> None:
    st.title("ComplaintCompass")
    st.caption(
        "Comparative NLP for automated consumer financial complaint routing"
    )
    st.warning(
        "Research prototype only. Do not use predictions as financial, legal, or "
        "regulatory advice."
    )
    artifacts_path = os.environ.get(
        "COMPLAINT_COMPASS_ARTIFACTS_DIR", str(ARTIFACTS_DIR)
    )
    try:
        predictor = load_predictor(artifacts_path)
    except ArtifactError as exc:
        st.error(str(exc))
        st.code(
            "python -m complaint_compass.data download\n"
            "python -m complaint_compass.data prepare\n"
            "python -m complaint_compass.train --all\n"
            "python -m complaint_compass.evaluate"
        )
        return

    classify_tab, compare_tab, about_tab = st.tabs(
        ["Classify complaint", "Compare models", "About the data"]
    )
    with classify_tab:
        _classify_tab(predictor)
    with compare_tab:
        _comparison_tab()
    with about_tab:
        _about_tab()


if __name__ == "__main__":
    main()
