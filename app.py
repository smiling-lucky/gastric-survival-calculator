from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src.inference import load_bundle, predict_case
from src.plots import plot_survival

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="Gastric Cancer Postoperative Survival Calculator",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def get_bundle():
    return load_bundle()


def default_inputs(bundle) -> dict:
    values = {}
    for spec in bundle["config"]["feature_meta"]:
        if spec["kind"] == "select":
            values[spec["internal"]] = int(spec["default"])
        else:
            values[spec["internal"]] = float(spec["default"])
    return values


def field_label(spec: dict) -> str:
    label = spec.get("label") or spec.get("display") or spec["internal"]
    if spec.get("unit"):
        return f"{label} ({spec['unit']})"
    return label


def render_number(spec: dict) -> float:
    observed_min = float(spec["observed_min"])
    observed_max = float(spec["observed_max"])
    span = max(observed_max - observed_min, 1e-6)
    return float(
        st.number_input(
            field_label(spec),
            step=float(spec["step"]),
            format="%.4f" if spec["step"] < 1 else "%.1f",
            help=spec["help"] + f" Training range: {observed_min:.4g} to {observed_max:.4g}.",
            key=spec["internal"],
            min_value=float(observed_min - 0.25 * span),
            max_value=float(observed_max + 0.25 * span),
        )
    )


bundle = get_bundle()
config = bundle["config"]

if "initialized" not in st.session_state:
    st.session_state.update(default_inputs(bundle))
    st.session_state.initialized = True

st.markdown(
    """
    <style>
    .result-card {padding: 1.1rem 1.2rem; border-radius: 12px; background: #ffffff;
                  box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 0.8rem;}
    .risk-low {color: #0072B2; font-weight: 700;}
    .risk-high {color: #D55E00; font-weight: 700;}
    .small-note {color: #555; font-size: 0.92rem; line-height: 1.5;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gastric Cancer Postoperative Multimodal Survival Calculator")
st.caption(
    "ExtraSurvivalTrees · 10 final variables · nested cross-validation winner (mean outer-fold Uno C-index 0.790)"
)

with st.expander("Read before use (academic research only)", expanded=True):
    st.markdown(
        """
- This tool is for academic research and methodological demonstration only. **It does not replace clinical decision-making.**
- **RS must be a precomputed radiomics score.** This app cannot read raw CT images or extract radiomic features.
- Training data contain **pTNM stages 2 and 3 only**, entered as numeric values (not one-hot encoded).
- 1-/3-/5-year event probabilities come from a Cox calibrator fit on full-cohort risk scores and are more optimistic than strict out-of-fold estimates.
- The high/low-risk cutoff is the median ExtraSurvivalTrees risk score in the training cohort. **It is not a clinical guideline cutoff.** Overlay KM curves use OOF z-score median grouping.
- No case data are saved or logged.
        """
    )

left, right = st.columns([0.92, 1.18], gap="large")

with left:
    st.subheader("Patient input")
    example_titles = {case["label"]: case["title"] for case in bundle["demo_cases"]}
    example_choice = st.selectbox(
        "Load an anonymized example (no patient ID)",
        options=["None"] + [case["label"] for case in bundle["demo_cases"]],
        format_func=lambda x: "None" if x == "None" else example_titles[x],
    )
    if st.button("Load selected example", width="stretch"):
        if example_choice != "None":
            case = next(item for item in bundle["demo_cases"] if item["label"] == example_choice)
            for name, value in case["features"].items():
                st.session_state[name] = int(value) if name == "pTNM" else float(value)
            st.rerun()

    with st.form("case_form"):
        columns = st.columns(2)
        collected = {}
        for index, spec in enumerate(config["feature_meta"]):
            with columns[index % 2]:
                if spec["kind"] == "select":
                    collected[spec["internal"]] = int(
                        st.selectbox(
                            field_label(spec),
                            options=[2, 3],
                            help=spec["help"],
                            key=spec["internal"],
                        )
                    )
                else:
                    collected[spec["internal"]] = render_number(spec)
        submitted = st.form_submit_button("Calculate prognosis", width="stretch")

with right:
    st.subheader("Prediction")
    if not submitted:
        st.info("Enter the variables on the left, then click Calculate prognosis.")
    else:
        with st.spinner("Computing risk score, calibrated probabilities, and survival curve..."):
            result = predict_case(collected, bundle)
        group = result["risk_group"]
        group_class = "risk-high" if group == "High risk" else "risk-low"
        p = result["event_probability"]
        st.markdown(
            f"""
            <div class="result-card">
              <h3 style="text-align:center;margin-bottom:0.4rem;">
                Risk group: <span class="{group_class}">{group}</span>
              </h3>
              <p style="text-align:center;margin:0.2rem 0;">Risk score (Risk_raw) = <b>{result['risk_raw']:.4f}</b></p>
              <p style="text-align:center;margin:0.2rem 0;" class="small-note">
                Cutoff = {result['cutoff']:.4f} (full-cohort final-model median; High risk if score ≥ cutoff)
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metrics = st.columns(3)
        for column, label, title in zip(
            metrics,
            ("1-year", "3-year", "5-year"),
            ("1-year event probability", "3-year event probability", "5-year event probability"),
        ):
            column.metric(title, f"{p[label]*100:.1f}%", help="1 − calibrated survival probability")
        st.caption(
            "Event probabilities come from the Cox calibrator and match the final-model prediction table. "
            "The curve is the native ExtraSurvivalTrees survival function."
        )

        fig = plot_survival(
            result["survival_curve"],
            bundle["km"],
            group,
            config["colors"],
            result["event_probability"],
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        out_of_range = []
        stats = config["feature_stats"]
        for name, value in collected.items():
            low, high = stats[name]["min"], stats[name]["max"]
            if value < low or value > high:
                display = config["display_name_map"].get(name, name)
                out_of_range.append(f"{display}={value:g} (training range {low:g}–{high:g})")
        if out_of_range:
            st.warning(
                "The following inputs are outside the training range; interpret with caution:\n\n- "
                + "\n- ".join(out_of_range)
            )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.85rem;'>"
    "Academic research use only · ExtraSurvivalTrees gastric postoperative survival model · no inputs are saved"
    "</div>",
    unsafe_allow_html=True,
)
