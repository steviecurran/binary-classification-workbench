from pathlib import Path
import sys
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Binary Classification Workbench",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stWidgetLabel"] p {
        color: #111111 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    [data-testid="stRadio"] label {
        color: #222222 !important;
        font-size: 16px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Binary Classification Workbench")

st.markdown(
    """
### How do model choice, class balance and decision thresholds change a binary prediction problem?

This interactive workbench grew from a **rare-event fraud-detection case study** into a reusable
binary-classification application. It supports built-in examples from fraud detection and published
astronomy research, as well as user-supplied datasets.

The emphasis is on **model comparison, class-imbalance handling, precision, recall, F1, threshold
selection and operational prediction rates** rather than relying on raw accuracy alone.
"""
)


# ---------------------------------------------------------------------
# Built-in datasets
# ---------------------------------------------------------------------

BUILTIN_DATASETS = {
    "Fraud detection — rare-event case study": {
        "file": "synthetic_gambling_aml.csv",
         "target": "is_suspicious_activity",
        "positive": 1,
        "exclude": [],
        "labels": {0: "Non-suspect", 1: "Suspect"},
        "imbalance_default": "Class weighting",
        "context": (
        "The case study contains 25,000 synthetic accounts with only about 1.5% "
        "positive/suspect cases, generated for this project (see DATA_DICTIONARY.md). "
        "It demonstrates why accuracy can be misleading when the event of interest is "
        "very rare."
    ),
        "paper_url": None,
        "paper_label": None,
        "fraud": True,
    },
    "Curran (2021) — H I absorption": {
        "file": "Curran_2021.csv",
        "target": "TYPE",
        "positive": "ass",
        "exclude": ["Name", "IAU", "ref", "cdda16", "Z"],
        "labels": {"ass": "Associated", "int": "Intervening"},
        "imbalance_default": "No adjustment",
        "context": (
            "Published astronomy example containing 136 redshifted H I 21-cm absorbers: "
            "80 associated and 56 intervening. The default predictors retain the spectral-profile "
            "features while excluding identifiers, provenance fields and redshift."
        ),
        "paper_url": "https://academic.oup.com/mnras/article/506/1/1548/6313314",
        "paper_label": "Curran (2021), MNRAS 506, 1548–1556",
        "fraud": False,
    },
    "Mondal et al. (2025) — H I absorption": {
        "file": "Mondal_2025.csv",
        "target": "Class",
        "positive": 1,
        "exclude": ["Name", "S", "z_abs", "SNR", "chi", "Reference "],
        "labels": {0: "Associated", 1: "Intervening"},
        "imbalance_default": "No adjustment",
        "context": (
            "Published astronomy example containing 118 H I 21-cm absorbers: 74 associated "
            "and 44 intervening. The paper labels associated as 0 and intervening as 1. "
            "The default predictor set here focuses on the 13 Busy-function spectral parameters."
        ),
        "paper_url": "https://academic.oup.com/mnras/article/544/4/3456/8315938",
        "paper_label": "Mondal et al. (2025), MNRAS 544, 3456–3471",
        "fraud": False,
    },
}

SOURCE_OPTIONS = [
    *BUILTIN_DATASETS.keys(),
    "Upload your own CSV",
    "Remote CSV URL",
]


# ---------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(source):
    return pd.read_csv(source)


def normalise_remote_url(url: str) -> str:
    cleaned = url.strip()
    marker = "drive.google.com/file/d/"
    if marker in cleaned:
        file_id = cleaned.split(marker, 1)[1].split("/", 1)[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return cleaned


def dataframe_memory_mb(dataframe: pd.DataFrame) -> float:
    return float(dataframe.memory_usage(index=True, deep=True).sum() / 1024**2)


def clean_text_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.map(lambda value: value.strip() if isinstance(value, str) else value)
    return series


def class_summary(target: pd.Series, positive_value) -> dict:
    positive = int((target == positive_value).sum())
    total = int(len(target))
    negative = total - positive
    rate = positive / total if total else np.nan
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "rate": rate,
    }


def prepare_xy(
    dataframe: pd.DataFrame,
    target_column: str,
    positive_value,
    excluded_columns: tuple[str, ...],
):
    working = dataframe.copy()
    y = (working[target_column] == positive_value).astype(int)

    candidate = working.drop(
        columns=[target_column, *excluded_columns],
        errors="ignore",
    )
    X = candidate.select_dtypes(include="number").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    constant_columns = [
        column for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        X = X.drop(columns=constant_columns)

    return X, y, constant_columns


# ---------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------

MODEL_NAMES = [
    "XGBoost",
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
]

IMBALANCE_METHODS = [
    "No adjustment",
    "Class weighting",
    "SMOTE",
    "Under-sampling",
]


def build_classifier(
    model_name: str,
    class_ratio: float,
    use_class_weighting: bool,
    random_seed: int,
):
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            n_jobs=2,
            eval_metric="logloss",
            scale_pos_weight=class_ratio if use_class_weighting else 1.0,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced" if use_class_weighting else None,
            random_state=random_seed,
            n_jobs=2,
        )

    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=0.08858667,
            solver="newton-cg",
            class_weight="balanced" if use_class_weighting else None,
            max_iter=2000,
            random_state=random_seed,
        )

    return DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced" if use_class_weighting else None,
        random_state=random_seed,
    )


@st.cache_resource(show_spinner=False)
def fit_and_score_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    imbalance_method: str,
    test_fraction: float,
    random_seed: int,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        stratify=y,
        random_state=random_seed,
    )

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    ratio = float(negatives / max(positives, 1))
    use_class_weighting = imbalance_method == "Class weighting"
    classifier = build_classifier(model_name, ratio, use_class_weighting, random_seed)

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]

    if imbalance_method == "SMOTE":
        minority_count = min(positives, negatives)
        if minority_count < 2:
            raise ValueError(
                "SMOTE requires at least two training observations in each class. "
                "Choose another imbalance method or increase the training fraction."
            )
        k_neighbors = min(5, minority_count - 1)
        steps.append(("sampler", SMOTE(random_state=random_seed, k_neighbors=k_neighbors)))

    elif imbalance_method == "Under-sampling":
        steps.append(
            (
                "sampler",
                RandomUnderSampler(
                    sampling_strategy="majority",
                    random_state=random_seed,
                ),
            )
        )

    steps.append(("classifier", classifier))
    model = Pipeline(steps)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    fitted_classifier = model.named_steps["classifier"]
    feature_importance = None

    if hasattr(fitted_classifier, "feature_importances_"):
        feature_importance = np.asarray(
            fitted_classifier.feature_importances_, dtype=float
        )
    elif hasattr(fitted_classifier, "coef_"):
        feature_importance = np.abs(
            np.asarray(fitted_classifier.coef_[0], dtype=float)
        )

    return {
        "model": model,
        "y_test": np.asarray(y_test),
        "y_prob": np.asarray(y_prob),
        "feature_names": tuple(X.columns),
        "feature_importance": feature_importance,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive": positives,
        "test_positive": int((y_test == 1).sum()),
    }


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positive_rate = float(y_pred.mean())
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "positive_rate": positive_rate,
        "positive_predictions": int(y_pred.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


@st.cache_data(show_spinner=False)
def threshold_table(y_true, y_prob):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        result = metrics_at_threshold(y_true, y_prob, float(threshold))
        rows.append(
            {
                "threshold": threshold,
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "positive_rate": result["positive_rate"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

def style_plot_axes(ax, plot_font, ax_width, grid=False):
    for spine in ax.spines.values():
        spine.set_linewidth(ax_width)

    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        pad=7, length=6, width=1.5, labelsize=plot_font,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True,
        length=3, width=1.2,
    )
    ax.minorticks_on()
    if grid:
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.8)

    return plot_font

def fix_log(axis, ticks, start, end):
    """Format symlog count ticks without Matplotlib math-text markup."""

    def update_ticks(value, pos):
        if not np.isfinite(value):
            return ""
        if np.isclose(value, 0):
            return "0"

        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:.0f}"
        return f"{value:g}"

    upper = max(float(start), float(end), 0.0)
    minor = []

    if upper >= 1:
        max_exponent = int(np.floor(np.log10(upper)))
        for exponent in range(max_exponent + 1):
            base = 10**exponent
            for multiplier in range(2, 10):
                tick = multiplier * base
                if tick <= upper:
                    minor.append(tick)

    if ticks == "xticks":
        axis.set_xticks(minor, minor=True)
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
    else:
        axis.set_yticks(minor, minor=True)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
        

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def make_class_histogram(
    dataframe,
    feature,
    y,
    bins,
    negative_label="Negative class",
    positive_label="Positive class",
):
    negative = pd.to_numeric(
        dataframe.loc[y == 0, feature], errors="coerce"
    ).dropna()
    positive = pd.to_numeric(
        dataframe.loc[y == 1, feature], errors="coerce"
    ).dropna()

    combined = pd.concat([negative, positive])
    if combined.empty:
        return None

    minimum = float(combined.min())
    maximum = float(combined.max())
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    edges = np.linspace(minimum, maximum, bins + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.hist(
        [negative, positive],
        bins=edges,
        color=["#4C78A8", "#E45756"],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label=[
            f"{negative_label} (negative, n={len(negative):,})",
            f"{positive_label} (positive, n={len(positive):,})",
        ],
    )
    ax.set_yscale("symlog")

    plot_font = style_plot_axes(ax, 13, 1.5)
    ax.set_xlabel(feature, size=plot_font)
    ax.set_ylabel("Number", size=plot_font)
    y1, y2 = ax.get_ylim()
    fix_log(ax, "yticks", y1, y2)

    ax.legend(fontsize=0.8 * plot_font, frameon=False)
    fig.tight_layout()
    return fig


def make_confusion_plot(result, negative_label, positive_label):
    matrix = np.array(
        [[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
        dtype=float,
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    labels = [["TN", "FP"], ["FN", "TP"]]

    for row in range(2):
        for col in range(2):
            cell_value = normalised[row, col]
            text_colour = "white" if cell_value >= 0.50 else "black"
            ax.text(
                col,
                row,
                f"{labels[row][col]}\n{int(matrix[row, col]):,}\n{cell_value:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_colour,
            )

    ax.set_xticks([0, 1], [negative_label, positive_label])
    ax.set_yticks([0, 1], [negative_label, positive_label])
    plot_font = style_plot_axes(ax, 17, 2)
    ax.set_xlabel("Predicted", fontsize=plot_font)
    ax.set_ylabel("Actual", fontsize=plot_font)
    ax.set_title("Confusion matrix", fontsize=plot_font, fontweight="normal", pad=12)
    ax.tick_params(axis="both", labelsize=plot_font)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.ax.tick_params(labelsize=plot_font)
    fig.tight_layout()
    return fig


def make_pr_plot(y_true, y_prob, threshold):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    result = metrics_at_threshold(y_true, y_prob, threshold)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot(
        recall,
        precision,
        color="#6F4E9C",
        linewidth=2.6,
        label="Precision-recall curve",
    )
    ax.scatter(
        result["recall"],
        result["precision"],
        s=90,
        color="#F28E2B",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label=f"Threshold = {threshold:.2f}",
    )
    plot_font = style_plot_axes(ax, 17, 2)
    ax.set_xlabel("Recall", size=plot_font)
    ax.set_ylabel("Precision", size=plot_font)
    ax.set_title(
        "Precision-recall curve",
        fontsize=plot_font,
        fontweight="normal",
        pad=12,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=0.8 * plot_font, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def make_roc_plot(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot(
        fpr,
        tpr,
        color="r",
        linewidth=2.6,
        label=f"ROC AUC = {roc_auc:.3f}",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="k",
        linestyle="dotted",
        linewidth=1.8,
        label="Random classifier",
    )
    plot_font = style_plot_axes(ax, 17, 2)
    ax.set_xlabel("False-positive rate", fontsize=plot_font)
    ax.set_ylabel("True-positive rate", fontsize=plot_font)
    ax.set_title(
        "Receiver operating characteristic",
        fontsize=plot_font,
        fontweight="normal",
        pad=12,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=0.8 * plot_font, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def make_probability_separation_plot(
    y_true,
    y_prob,
    confidence_percent,
    negative_label,
    positive_label,
):
    positive = y_prob[y_true == 1]
    negative = y_prob[y_true == 0]
    z_value = norm.ppf(0.5 + confidence_percent / 200.0)

    def mean_ci(values):
        mean = float(np.mean(values))
        if len(values) < 2:
            return mean, 0.0
        se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        return mean, z_value * se

    positive_mean, positive_ci = mean_ci(positive)
    negative_mean, negative_ci = mean_ci(negative)

    fig, ax = plt.subplots(figsize=(6.3, 2.45))
    ax.errorbar(
        positive_mean,
        0,
        xerr=positive_ci,
        fmt="o",
        markersize=9,
        color="#E45756",
        ecolor="#E45756",
        elinewidth=2.2,
        capsize=6,
        capthick=1.8,
        markeredgecolor="black",
        markeredgewidth=0.7,
        label=positive_label,
    )
    ax.errorbar(
        negative_mean,
        1,
        xerr=negative_ci,
        fmt="o",
        markersize=9,
        color="#4C78A8",
        ecolor="#4C78A8",
        elinewidth=2.2,
        capsize=6,
        capthick=1.8,
        markeredgecolor="black",
        markeredgewidth=0.7,
        label=negative_label,
    )

    plot_font = style_plot_axes(ax, 17, 2)
    ax.set_xlabel(
        f"Mean predicted positive-class probability ({confidence_percent:.1f}% CI)",
        fontsize=plot_font,
    )
    ax.set_title(
        "Predicted-probability separation",
        size=plot_font,
        fontweight="normal",
        pad=7,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.75, 1.75)

    ax.set_yticks([])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    ax.legend(fontsize=0.8 * plot_font, frameon=False, loc="best", ncol=1)
    fig.tight_layout(pad=0.8)
    return fig


def make_threshold_plot(table, selected_threshold, rate_label):
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.plot(
        table["threshold"], table["f1"],
        color="dimgrey", lw=2, label="F1 score", zorder=2,
    )
    ax.plot(
        table["threshold"], table["precision"],
        color="b", ls="--", lw=2, label="Precision",
    )
    ax.plot(
        table["threshold"], table["recall"],
        color="lime", ls="--", lw=2, label="Recall",
    )
    ax.plot(
        table["threshold"], table["positive_rate"],
        color="orange", ls="--", lw=2, label=rate_label,
    )
    ax.axvline(
        selected_threshold,
        color="r",
        ls="dotted",
        lw=2,
        label="Selected threshold",
    )

    plot_font = style_plot_axes(ax, 13, 1.5)
    ax.set_xlabel("Classification threshold", size=plot_font)
    ax.set_ylabel("Metric value", size=plot_font)
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=0.8 * plot_font, ncol=2)
    fig.tight_layout()
    return fig


def make_feature_importance_plot(feature_names, importance):
    frame = pd.DataFrame(
        {"Feature": list(feature_names), "Importance": importance}
    ).sort_values("Importance", ascending=True)

    fig_height = max(4.0, 0.3 * len(frame))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(
        frame["Feature"],
        frame["Importance"],
        color="silver",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )
    plot_font = style_plot_axes(ax, 10, 1.0)
    ax.tick_params(
        axis="both", which="major", direction="in", top=False, right=False,
        pad=7, length=3, width=1.5, labelsize=plot_font,
    )
    ax.minorticks_off()
    ax.set_xlabel("Importance", size=plot_font)
    fig.tight_layout()
    return fig, frame.sort_values("Importance", ascending=False)


# ---------------------------------------------------------------------
# Sidebar data source
# ---------------------------------------------------------------------


with st.sidebar:

    st.header("Data source")

    source_group = st.radio(
        "Dataset group",
        [
            "Business-focused classification",
            "Absorbing-galaxy classification",
            "Use your own data",
        ],
    )

    if source_group == "Business-focused classification":
        source_type = "Fraud detection — rare-event case study"
        st.caption("Fraud detection — rare-event case study")

    elif source_group == "Absorbing-galaxy classification":
        astronomy_example = st.radio(
            "Published example",
            [
                "Curran (2021)",
                "Mondal et al. (2025)",
            ],
        )

        if astronomy_example == "Curran (2021)":
            source_type = "Curran (2021) — H I absorption"
        else:
            source_type = "Mondal et al. (2025) — H I absorption"

    else:
        source_type = st.radio(
            "Your data source",
            [
                "Upload your own CSV",
                "Remote CSV URL",
            ],
        )

    data = None
    source_label = None
    source_config = BUILTIN_DATASETS.get(source_type)

    if source_config is not None:
        built_in_path = DATA_DIR / source_config["file"]
        if built_in_path.exists():
            data = load_csv(built_in_path)
            source_label = source_type
        else:
            st.error(f"Could not find `{built_in_path.name}` in the data directory.")

    elif source_type == "Upload your own CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
                source_label = f"Uploaded file: {uploaded.name}"
            except Exception as error:
                st.error(f"Could not read the uploaded file. {error}")

    else:
        remote_url = st.text_input(
            "Public CSV URL",
            placeholder="Paste a Google Drive or other public CSV URL",
        )
        if remote_url:
            try:
                data = load_csv(normalise_remote_url(remote_url))
                source_label = "Remote CSV"
            except Exception as error:
                st.error(f"Could not load the remote CSV. {error}")

    if data is not None:
        st.markdown("### Loaded dataset")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(data):,}")
        c2.metric("Memory", f"{dataframe_memory_mb(data):.1f} MB")
        st.caption(source_label or "Loaded dataset")


if data is None:
    st.info("Choose or load a dataset from the sidebar to begin.")
    st.stop()


# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

all_columns = data.columns.tolist()
st.sidebar.divider()
st.sidebar.subheader("Configure classification")



if source_config is not None:
    target_column = source_config["target"]
    st.sidebar.text_input("Target column", value=target_column, disabled=True)
else:
    target_column = st.sidebar.selectbox(
        "Target column",
        all_columns,
        index=(
            all_columns.index("is_suspicious_activity")
            if "is_suspicious_activity" in all_columns
            else len(all_columns) - 1
        ),
    )

# Remove accidental whitespace from string target values before checking classes.
data[target_column] = clean_text_target(data[target_column])
target_values = data[target_column].dropna().unique().tolist()

if len(target_values) != 2:
    st.error(
        "The selected target must contain exactly two classes for this binary "
        "classification workbench."
    )
    st.stop()

if source_config is not None and source_config["positive"] in target_values:
    positive_default_index = target_values.index(source_config["positive"])
else:
    positive_default_index = 1 if len(target_values) > 1 else 0

positive_value = st.sidebar.selectbox(
    "Positive class",
    target_values,
    index=positive_default_index,
    help=(
        "Precision, recall, F1 and predicted probabilities are calculated with "
        "respect to this class."
    ),
)

negative_values = [value for value in target_values if value != positive_value]
negative_value = negative_values[0]

if source_config is not None:
    label_map = source_config["labels"]
    negative_class_label = str(label_map.get(negative_value, negative_value))
    positive_class_label = str(label_map.get(positive_value, positive_value))
    excluded_default = [
        column for column in source_config["exclude"] if column in all_columns
    ]
else:
    negative_class_label = str(negative_value)
    positive_class_label = str(positive_value)
    identifier_guess = [
        column for column in all_columns
        if column.lower().strip() in {"id", "index", "name"}
    ]
    excluded_default = identifier_guess[:1]

excluded_columns = st.sidebar.multiselect(
    "Exclude identifier / non-predictor columns",
    [column for column in all_columns if column != target_column],
    default=excluded_default,
)

X, y, constant_columns = prepare_xy(
    data,
    target_column,
    positive_value,
    tuple(excluded_columns),
)

if X.empty:
    st.error("No usable numeric predictor columns remain after configuration.")
    st.stop()

summary = class_summary(data[target_column], positive_value)

is_fraud_case = bool(source_config and source_config.get("fraud"))
rate_label = "Alert rate" if is_fraud_case else "Positive prediction rate"


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------

tabs = st.tabs(
    [
        "Dataset overview",
        "Feature Explorer",
        "Model Workbench",
        "Threshold Tuning",
        "Interpretation",
        "Predict New Data",
    ]
)


# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------

with tabs[0]:
    st.subheader("Dataset overview")

    if source_config is not None:
        st.info(source_config["context"])
        if source_config["paper_url"]:
            st.markdown(
                f"**Source paper:** [{source_config['paper_label']}]"
                f"({source_config['paper_url']})"
            )
            st.caption(
                "The workbench uses its own configurable train/test workflow; results shown here "
                "should not be interpreted as a reproduction of the paper's published validation protocol."
            )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{summary['total']:,}")
    m2.metric(f"{positive_class_label} / positive", f"{summary['positive']:,}")
    m3.metric(f"{negative_class_label} / negative", f"{summary['negative']:,}")
    m4.metric("Positive-class rate", f"{100 * summary['rate']:.2f}%")

    majority_is_positive = summary["positive"] >= summary["negative"]
    majority_label = positive_class_label if majority_is_positive else negative_class_label
    majority_accuracy = max(summary["rate"], 1 - summary["rate"])

    if is_fraud_case:
        st.markdown(
            f"""
If every case were simply predicted as **{negative_class_label}**, the model would appear to achieve
**{100 * (1 - summary['rate']):.2f}% accuracy** while detecting **none** of the positive cases.
That is why this project evaluates precision, recall, F1 and alert workload instead of optimising raw accuracy.
"""
        )
    else:
        st.markdown(
            f"""
A classifier that predicts the majority class (**{majority_label}**) for every row would achieve
**{100 * majority_accuracy:.2f}% accuracy** without learning any useful separation between the classes.
The remaining tabs therefore focus on class-specific and threshold-sensitive evaluation.
"""
        )

    if constant_columns:
        st.caption(
            "Constant columns automatically excluded from modelling: "
            + ", ".join(constant_columns)
        )

    with st.expander("Preview data"):
        total_rows = len(data)
        preview_cap = 20_000

        st.markdown("**Choose number of rows to display**")

        if total_rows <= 10:
            preview_rows = total_rows
            st.caption(f"Showing all {total_rows:,} rows.")
        else:
            slider_max = min(total_rows, preview_cap)
            row_options = np.geomspace(10, slider_max, num=50)
            row_options = sorted(set(int(round(value)) for value in row_options))

            if slider_max not in row_options:
                row_options.append(slider_max)

            default_rows = min(100, slider_max)
            default_value = min(
                row_options,
                key=lambda value: abs(value - default_rows),
            )

            preview_rows = st.select_slider(
                "Rows to preview",
                options=row_options,
                value=default_value,
                format_func=lambda value: f"{value:,}",
                label_visibility="collapsed",
            )

            if total_rows > preview_cap:
                show_all = st.checkbox(f"Show all {total_rows:,} rows")
                if show_all:
                    preview_rows = total_rows

            st.caption(f"Showing {preview_rows:,} of {total_rows:,} rows.")

        st.dataframe(
            data.head(preview_rows),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Descriptive statistics"):
        st.dataframe(X.describe().T, use_container_width=True)


# ---------------------------------------------------------------------
# Feature explorer
# ---------------------------------------------------------------------

with tabs[1]:
    st.subheader("Feature Explorer")
    st.markdown(
        "Compare the distribution of any numeric predictor between the selected positive and "
        "negative classes. The y-axis uses a symlog scale so both classes remain readable when "
        "their sample sizes differ substantially."
    )

    f1, f2 = st.columns([1, 1])
    with f1:
        feature = st.selectbox("Feature", list(X.columns), index=0)
    with f2:
        number_of_bins = st.select_slider(
            "Number of histogram bins",
            options=[5, 10, 20, 30, 50, 75, 100],
            value=30,
        )

    explorer = pd.DataFrame({feature: X[feature], "_class": y})
    figure = make_class_histogram(
        explorer,
        feature,
        explorer["_class"],
        number_of_bins,
        negative_class_label,
        positive_class_label,
    )
    if figure is not None:
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    stats_rows = []
    for class_value, label in [
        (0, negative_class_label),
        (1, positive_class_label),
    ]:
        values = X.loc[y == class_value, feature]
        stats_rows.append(
            {
                "Class": label,
                "n": len(values),
                "Mean": values.mean(),
                "Std. dev.": values.std(ddof=1),
                "Median": values.median(),
            }
        )
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Model Workbench
# ---------------------------------------------------------------------

with tabs[2]:
    st.subheader("Model Workbench")
    st.markdown(
        "Fit a classifier using a stratified hold-out split. Imputation, scaling and any "
        "resampling are learned from the training data only; the held-out test set retains its "
        "original class distribution."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        model_name = st.selectbox(
            "Classifier",
            MODEL_NAMES,
            index=0,
            key="model_name",
        )
        with st.popover("ℹ️ About classifiers"):
            st.markdown(
                """
**XGBoost**  
Builds decision trees sequentially so each stage corrects errors made earlier. Often strong on structured tabular data and capable of modelling complex non-linear relationships.

**Random Forest**  
Fits many decision trees to different bootstrap samples and combines their predictions. Robust, non-linear and particularly useful for tabular data.

**Logistic Regression**  
An interpretable linear classifier that estimates positive-class probability. Useful as a strong baseline and when transparency matters.

**Decision Tree**  
Creates rule-based splits through the predictor space. Easy to interpret, although a single tree can overfit more readily than an ensemble.
"""
            )

    with c2:
        imbalance_default = (
            source_config["imbalance_default"]
            if source_config is not None
            else "No adjustment"
        )
        imbalance_method = st.selectbox(
            "Class-imbalance method",
            IMBALANCE_METHODS,
            index=IMBALANCE_METHODS.index(imbalance_default),
            key="imbalance_method",
        )
        with st.popover("ℹ️ Class-imbalance"):
            st.markdown(
                """
**No adjustment**  
Uses the observed training data as-is. A sensible starting point when class proportions are not strongly imbalanced.

**Class weighting**  
Gives greater importance to the minority class during model training without changing the observations.

**SMOTE**  
Creates synthetic minority-class examples within the training data to improve representation of the less common class.

**Under-sampling**  
Reduces the number of majority-class training observations to create a more balanced training set.
"""
            )

    with c3:
        test_fraction = st.select_slider(
            "Test fraction",
            options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            value=0.20,
            key="test_fraction",
        )

    with c4:
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999_999,
            value=42,
            step=1,
            key="random_seed",
            help=(
                "Controls the random shuffle used for the stratified train/test split and "
                "the stochastic parts of model fitting or resampling. The same seed gives "
                "the same result; changing it produces a different reproducible run."
            ),
        )

        with st.popover("ℹ️ Random seed"):
            st.markdown(
                """
    The data **are shuffled** before the stratified train/test split. The seed simply makes that
    shuffle reproducible.

    - **Seed 42 is not special.** It is just a conventional fixed starting value.
    - The same seed gives the same train/test split and random model choices.
    - Changing the seed creates a different, but still reproducible, run.
    - This is particularly useful for **small datasets**, where a few observations moving between
      training and test sets can noticeably change precision, recall or F1.

    Try several seeds to assess **split sensitivity**, but do not choose the seed simply because it
    gives the best result. For formal performance estimation, repeated stratified cross-validation
    is preferable.

    The seed is a reproducibility control, **not a model-tuning parameter**.
    """
            )

    if model_name == "XGBoost" and is_fraud_case:
        st.caption(
            "For the fraud example, the XGBoost settings match the original notebook case study: "
            "500 estimators, depth 5 and learning rate 0.03."
        )

    if (
        source_type == "Mondal et al. (2025) — H I absorption"
        and model_name == "Random Forest"
    ):
        st.caption(
            "Mondal et al. (2025) reported Random Forest as their strongest classifier under a "
            "different repeated evaluation protocol. Results in this workbench are not a direct reproduction."
        )

    st.caption(
        f"Current split seed: **{int(random_seed)}**. Re-running unchanged settings is reproducible; "
        "change the seed to generate a different stratified shuffle."
    )

    run_model = st.button("Run model", type="primary")

    config_signature = (
        source_label,
        target_column,
        str(positive_value),
        model_name,
        imbalance_method,
        test_fraction,
        int(random_seed),
        tuple(X.columns),
        len(X),
        int(y.sum()),
    )

    if run_model:
        with st.spinner("Fitting model and scoring the untouched test set..."):
            try:
                result = fit_and_score_model(
                    X,
                    y,
                    model_name,
                    imbalance_method,
                    test_fraction,
                    int(random_seed),
                )
                st.session_state["binary_result"] = result
                st.session_state["binary_signature"] = config_signature
            except Exception as error:
                st.exception(error)

    result = st.session_state.get("binary_result")
    stored_signature = st.session_state.get("binary_signature")
    current_result_available = result is not None and stored_signature == config_signature

    if not current_result_available:
        if result is not None and stored_signature != config_signature:
            st.warning(
                "The dataset or model controls have changed since the displayed model was fitted. "
                "Click **Run model** to fit the current configuration."
            )
        else:
            st.info("Choose the model settings and click **Run model**.")
    else:
        workbench_thresholds = threshold_table(
            result["y_test"],
            result["y_prob"],
        )
        best_workbench_row = workbench_thresholds.loc[
            workbench_thresholds["f1"].idxmax()
        ]
        best_workbench_threshold = float(
            round(best_workbench_row["threshold"], 2)
        )

        threshold_state_key = "binary_model_threshold"
        threshold_signature_key = "binary_model_threshold_signature"

        if st.session_state.get(threshold_signature_key) != stored_signature:
            st.session_state[threshold_state_key] = best_workbench_threshold
            st.session_state[threshold_signature_key] = stored_signature

        decision_col, threshold_note_col = st.columns(2)

        with decision_col:
            threshold = st.slider(
                "Classification threshold",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key=threshold_state_key,
                help=(
                    "The slider starts at the threshold that maximises F1 for the current fitted model. "
                    "Moving it changes the classification decision without retraining the model."
                ),
            )
            st.caption(
                f"The default threshold ({best_workbench_threshold:.2f}) is selected by sweeping "
                "candidate thresholds from 0.01 to 0.99 and choosing the value that maximises F1. "
                "The full precision–recall trade-off can be explored in Threshold Tuning."
            )

        with threshold_note_col:
            st.markdown("**How to read the threshold**")
            st.caption(
                "A higher threshold usually produces fewer positive predictions and fewer false positives, "
                "but can miss more true positives. A lower threshold usually increases recall at the cost "
                "of more positive predictions."
            )

        current = metrics_at_threshold(
            result["y_test"],
            result["y_prob"],
            threshold,
        )

        a, b, c, d = st.columns(4)
        a.metric("Precision", f"{current['precision']:.3f}")
        b.metric("Recall", f"{current['recall']:.3f}")
        c.metric("F1 score", f"{current['f1']:.3f}")
        d.metric(rate_label, f"{100 * current['positive_rate']:.2f}%")

        st.caption(
            f"Test set: {result['test_rows']:,} observations, including "
            f"{result['test_positive']:,} positive cases. At this threshold, "
            f"{current['positive_predictions']:,} observations are classified as {positive_class_label}."
        )

        left, right = st.columns(2)
        with left:
            fig = make_confusion_plot(
                current,
                negative_class_label,
                positive_class_label,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            fig = make_pr_plot(result["y_test"], result["y_prob"], threshold)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        left, right = st.columns(2)
        with left:
            fig = make_roc_plot(result["y_test"], result["y_prob"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            confidence = st.session_state.get("binary_probability_ci", 95.0)
            fig = make_probability_separation_plot(
                result["y_test"],
                result["y_prob"],
                confidence,
                negative_class_label,
                positive_class_label,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.select_slider(
                "Probability mean confidence interval",
                options=[80.0, 90.0, 95.0, 99.0, 99.9, 99.99, 99.999],
                value=confidence,
                key="binary_probability_ci",
                help=(
                    "Controls the uncertainty interval around the mean predicted probability for each class. "
                    "It does not change the classifier or its predictions."
                ),
            )
            st.caption(
                "The points show the mean predicted positive-class probability for the two actual classes; "
                "the error bars show uncertainty around each mean."
            )

        if result["feature_importance"] is not None:
            with st.expander("Feature importance"):
                fig, importance_table = make_feature_importance_plot(
                    result["feature_names"],
                    result["feature_importance"],
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(
                    importance_table,
                    use_container_width=True,
                    hide_index=True,
                )


# ---------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------

with tabs[3]:
    st.subheader("Threshold Tuning")
    st.markdown(
        "The classification threshold is an operational decision, not just a model parameter. "
        "Increasing it generally reduces false positives and the positive prediction rate, but can "
        "also allow more true positives to go undetected. "
        "**F1 combines precision and recall, rewarding models that perform well on both.**"
    )

    result = st.session_state.get("binary_result")
    stored_signature = st.session_state.get("binary_signature")
    current_result_available = result is not None and stored_signature == config_signature

    table = None
    if current_result_available:
        table = threshold_table(result["y_test"], result["y_prob"])
    elif is_fraud_case:
        precomputed_path = DATA_DIR / "fraud_thresholds.csv"
        if precomputed_path.exists():
            raw_thresholds = load_csv(precomputed_path)
            rename_map = {
                "f-score": "f1",
                "alert": "positive_rate",
                "alert_rate": "positive_rate",
            }
            table = raw_thresholds.rename(columns=rename_map)
            required = ["threshold", "f1", "precision", "recall", "positive_rate"]
            if all(column in table.columns for column in required):
                table = table[required]
                st.info(
                    "Showing the precomputed XGBoost + class-weighting threshold sweep from the original "
                    "fraud notebook. Run a model in Model Workbench to generate a live sweep."
                )
            else:
                table = None

    if table is None:
        st.info("Run the current configuration in **Model Workbench** to explore its threshold trade-offs.")
    else:
        best_row = table.loc[table["f1"].idxmax()]
        threshold_choice = st.slider(
            "Inspect threshold",
            min_value=0.01,
            max_value=0.99,
            value=float(round(best_row["threshold"], 2)),
            step=0.01,
            key="binary_threshold_tuning_slider",
        )

        nearest_index = (table["threshold"] - threshold_choice).abs().idxmin()
        selected = table.loc[nearest_index]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{selected['precision']:.3f}")
        m2.metric("Recall", f"{selected['recall']:.3f}")
        m3.metric("F1 score", f"{selected['f1']:.3f}")
        m4.metric(rate_label, f"{100 * selected['positive_rate']:.2f}%")

        st.markdown(
            f"""
            <div style="font-size: 20px; font-weight: 600;">
            Maximum F1 = {best_row['f1']:.3f} at threshold {best_row['threshold']:.2f}.
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = make_threshold_plot(table, threshold_choice, rate_label)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Threshold table"):
            display_table = table.copy()
            display_table["positive_rate"] *= 100
            display_table = display_table.rename(
                columns={"positive_rate": "positive_prediction_rate_percent"}
            )
            st.dataframe(display_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------

with tabs[4]:
    st.subheader("Interpretation")

    result = st.session_state.get("binary_result")
    stored_signature = st.session_state.get("binary_signature")
    current_result_available = result is not None and stored_signature == config_signature

    st.markdown(
        f"""
The selected positive class is **{positive_class_label}**, representing
**{summary['positive']:,} of {summary['total']:,} observations ({100 * summary['rate']:.2f}%)**.

The useful question is not simply whether a classifier is accurate, but how effectively it separates
**{positive_class_label}** from **{negative_class_label}** at a decision threshold that makes sense for the task.
"""
    )

    if current_result_available:
        table = threshold_table(result["y_test"], result["y_prob"])
        best = table.loc[table["f1"].idxmax()]
        best_metrics = metrics_at_threshold(
            result["y_test"],
            result["y_prob"],
            float(best["threshold"]),
        )

        st.markdown(
            f"""
For the currently fitted model, the threshold that maximises F1 is approximately
**{best['threshold']:.2f}**, producing **{best['precision']:.1%} precision**,
**{best['recall']:.1%} recall**, **F1 = {best['f1']:.3f}**, and a
**{best['positive_rate']:.2%} positive prediction rate** on the held-out test set.

That corresponds to **{best_metrics['positive_predictions']:,} positive predictions** from
**{result['test_rows']:,} test observations**.
"""
        )

        if result["test_rows"] < 100:
            st.warning(
                "The held-out test set is small, so individual observations can materially change the reported metrics. "
                "For formal inference, repeated resampling or cross-validation would provide more stable estimates."
            )

    elif is_fraud_case:
        st.markdown(
            """
In the original fraud notebook, **XGBoost with class weighting** provided a useful operating point
near a threshold of **0.65**, with roughly **20.0% precision**, **25.3% recall**, **F1 ≈ 0.224** and an
alert rate of only about **1.90%**. The point of the workbench is to make that trade-off explicit and adjustable.
"""
        )
    elif source_config is not None and source_config["paper_url"]:
        st.markdown(
            f"The selected dataset is drawn from [{source_config['paper_label']}]({source_config['paper_url']}). "
            "Run a model in **Model Workbench** to evaluate it using this application's hold-out workflow."
        )

    st.caption(
        "This application is an analytical demonstration. Model performance depends on the data-generating process, "
        "feature quality, class prevalence and the relative costs of false positives and false negatives."
    )


# ---------------------------------------------------------------------
# Predict new / unlabelled data
# ---------------------------------------------------------------------

with tabs[5]:
    st.subheader("Predict New Data")
    st.markdown(
        "Upload a fresh, unlabelled CSV and score it using the **currently fitted model**. "
        "The app applies the same training-time imputation and scaling pipeline, then converts "
        "predicted probabilities into binary classes using the selected decision threshold."
    )

    result = st.session_state.get("binary_result")
    stored_signature = st.session_state.get("binary_signature")
    current_result_available = result is not None and stored_signature == config_signature

    if not current_result_available:
        st.info(
            "Fit the current dataset and model configuration in **Model Workbench** before scoring new data."
        )
    else:
        required_features = list(result["feature_names"])
        fitted_table = threshold_table(result["y_test"], result["y_prob"])
        fitted_best = fitted_table.loc[fitted_table["f1"].idxmax()]
        default_prediction_threshold = float(
            st.session_state.get(
                "binary_model_threshold",
                round(float(fitted_best["threshold"]), 2),
            )
        )

        prediction_threshold_signature_key = "prediction_threshold_signature"
        if st.session_state.get(prediction_threshold_signature_key) != stored_signature:
            st.session_state["prediction_threshold"] = default_prediction_threshold
            st.session_state[prediction_threshold_signature_key] = stored_signature

        p1, p2 = st.columns([1, 1])
        with p1:
            prediction_threshold = st.slider(
                "Prediction threshold",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key="prediction_threshold",
                help=(
                    "Initialised from the threshold currently selected in Model Workbench. "
                    "You can change it here without retraining the model."
                ),
            )

            st.caption(
                f"The current default ({default_prediction_threshold:.2f}) comes from the fitted model's "
                "validation behaviour, not from an assumption that binary classification should use 0.50."
            )

        with p2:
            st.markdown("**Required predictor schema**")
            st.caption(
                f"{len(required_features)} numeric predictor columns are required. Extra columns are retained "
                "in the output but ignored by the model."
            )

        with st.expander("Why can the threshold be far from 0.50?"):
            st.markdown(
                """
**0.50 is a conventional probability cut-off, not a universally optimal classification threshold.**
It simply says: classify an observation as positive when the model assigns a probability of at least 50%.

The best operating threshold can be much lower or higher because:

- **The classes may be imbalanced.** A rare positive class can require a lower threshold to recover useful numbers of true positives.
- **False positives and false negatives may have different consequences.** Lowering the threshold usually increases recall but also produces more positive predictions; raising it usually improves selectivity at the cost of missed positives.
- **Model probabilities are not necessarily perfectly calibrated.** A score of 0.15 can still be a very strong positive signal relative to the scores produced for the negative class.
- **The objective matters.** This workbench initially selects the threshold that maximises **F1** on the held-out test set, balancing precision and recall. A different application could instead choose a threshold for a required recall, precision, workload or cost.

So a threshold such as **0.15 is not saying that a 15% event probability is intrinsically 'more likely than not'.**
It is saying that, for this fitted model and the chosen decision objective, observations scoring at least 0.15 are best treated as positive.

Use **0.50** when that conventional cut-off is appropriate for the application, or when probabilities are well calibrated and the costs of the two error types are roughly symmetric. Otherwise the threshold should be treated as an explicit decision rule.
"""
            )

            selected_validation = metrics_at_threshold(
                result["y_test"],
                result["y_prob"],
                prediction_threshold,
            )
            half_validation = metrics_at_threshold(
                result["y_test"],
                result["y_prob"],
                0.50,
            )

            validation_comparison = pd.DataFrame(
                [
                    {
                        "Threshold": f"{prediction_threshold:.2f} (selected)",
                        "Precision": selected_validation["precision"],
                        "Recall": selected_validation["recall"],
                        "F1": selected_validation["f1"],
                        "Positive prediction rate": selected_validation["positive_rate"],
                    },
                    {
                        "Threshold": "0.50 (reference)",
                        "Precision": half_validation["precision"],
                        "Recall": half_validation["recall"],
                        "F1": half_validation["f1"],
                        "Positive prediction rate": half_validation["positive_rate"],
                    },
                ]
            )

            st.markdown("**Held-out validation comparison**")
            st.dataframe(
                validation_comparison.style.format(
                    {
                        "Precision": "{:.3f}",
                        "Recall": "{:.3f}",
                        "F1": "{:.3f}",
                        "Positive prediction rate": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "This comparison uses the held-out labelled test set from Model Workbench. "
                "It helps explain the decision threshold before it is applied to fresh, unlabelled data."
            )

        with st.expander("Required predictor columns"):
            schema = pd.DataFrame(
                {
                    "Predictor": required_features,
                    "Training dtype": [str(X[column].dtype) for column in required_features],
                }
            )
            st.dataframe(schema, use_container_width=True, hide_index=True)

            template = pd.DataFrame(columns=required_features)
            st.download_button(
                "Download empty input template",
                data=template.to_csv(index=False).encode("utf-8"),
                file_name="binary_prediction_input_template.csv",
                mime="text/csv",
            )

        prediction_file = st.file_uploader(
            "Upload unlabelled CSV for prediction",
            type=["csv"],
            key="prediction_file",
        )

        if prediction_file is not None:
            try:
                prediction_data = pd.read_csv(prediction_file)
            except Exception as error:
                st.error(f"Could not read the prediction CSV. {error}")
                prediction_data = None

            if prediction_data is not None:
                missing_columns = [
                    column for column in required_features
                    if column not in prediction_data.columns
                ]

                if missing_columns:
                    st.error(
                        "The prediction file is missing required predictor columns: "
                        + ", ".join(missing_columns)
                    )
                else:
                    prediction_X = prediction_data[required_features].copy()
                    prediction_X = prediction_X.apply(pd.to_numeric, errors="coerce")
                    prediction_X = prediction_X.replace([np.inf, -np.inf], np.nan)

                    try:
                        predicted_probability = result["model"].predict_proba(prediction_X)[:, 1]
                        predicted_binary = (
                            predicted_probability >= prediction_threshold
                        ).astype(int)

                        scored = prediction_data.copy()
                        scored["Predicted positive probability"] = predicted_probability
                        scored["Predicted class"] = np.where(
                            predicted_binary == 1,
                            positive_class_label,
                            negative_class_label,
                        )

                        positive_predictions = int(predicted_binary.sum())
                        total_predictions = int(len(predicted_binary))
                        positive_prediction_rate = (
                            positive_predictions / total_predictions
                            if total_predictions
                            else np.nan
                        )

                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Rows scored", f"{total_predictions:,}")
                        r2.metric(
                            f"Predicted {positive_class_label}",
                            f"{positive_predictions:,}",
                        )
                        r3.metric(
                            f"Predicted {negative_class_label}",
                            f"{total_predictions - positive_predictions:,}",
                        )
                        r4.metric(
                            "Positive prediction rate",
                            f"{100 * positive_prediction_rate:.2f}%"
                            if total_predictions
                            else "—",
                        )

                        st.caption(
                            f"Predictions use threshold {prediction_threshold:.2f}. These are model outputs for "
                            "unlabelled observations, not validation metrics."
                        )

                        st.dataframe(
                            scored.head(200),
                            use_container_width=True,
                            hide_index=True,
                        )
                        if len(scored) > 200:
                            st.caption(f"Previewing 200 of {len(scored):,} scored rows.")

                        st.download_button(
                            "Download scored CSV",
                            data=scored.to_csv(index=False).encode("utf-8"),
                            file_name="binary_classification_predictions.csv",
                            mime="text/csv",
                            type="primary",
                        )
                    except Exception as error:
                        st.exception(error)
