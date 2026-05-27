import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="QuakeAnalyze", page_icon="🌎", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_earthquakes.csv")


@st.cache_resource
def train_model(data):
    feature_cols = ["magnitude", "depth", "cdi", "mmi", "sig"]
    target_col = "risk_level"

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return model, feature_cols, X_train, X_test, y_test, y_pred, accuracy, report, labels, cm


df = load_data()

df["risk_level"] = df["alert"].replace({
    "green": "low",
    "yellow": "medium",
    "orange": "medium",
    "red": "high"
})

st.title("🌎 QuakeAnalyze")
st.write("Earthquake Risk Analysis Dashboard with Logistic Regression")

required_columns = ["magnitude", "depth", "cdi", "mmi", "sig", "alert"]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

model, feature_cols, X_train, X_test, y_test, y_pred, accuracy, report, labels, cm = train_model(df)

eda_tab, model_tab, predict_tab = st.tabs(
    ["📊 EDA Dashboard", "🤖 Logistic Regression Model", "🔮 Predict Risk"]
)


with eda_tab:
    st.subheader("Dataset Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Average Magnitude", round(df["magnitude"].mean(), 2))
    col4.metric("Most Common Alert", df["alert"].mode()[0])

    st.sidebar.header("Filters")

    min_mag = st.sidebar.slider(
        "Minimum Magnitude",
        float(df["magnitude"].min()),
        float(df["magnitude"].max()),
        float(df["magnitude"].min())
    )

    selected_alerts = st.sidebar.multiselect(
        "Alert Levels",
        options=sorted(df["alert"].unique()),
        default=sorted(df["alert"].unique())
    )

    filtered_df = df[
        (df["magnitude"] >= min_mag) &
        (df["alert"].isin(selected_alerts))
    ]

    st.write(f"Showing **{len(filtered_df):,}** earthquakes after filtering.")

    fig1 = px.histogram(
        filtered_df,
        x="magnitude",
        color="alert",
        title="Magnitude Distribution by Alert Level",
        nbins=30
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Make a safe positive column for Plotly marker size
    filtered_df = filtered_df.copy()
    filtered_df["sig_size"] = filtered_df["sig"].abs()

    fig2 = px.scatter(
        filtered_df,
        x="depth",
        y="magnitude",
        color="alert",
        size="sig_size",
        hover_data=["cdi", "mmi", "sig"],
        title="Depth vs Magnitude"
        )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        filtered_df["alert"].value_counts().reset_index(),
        x="alert",
        y="count",
        title="Number of Earthquakes by Alert Level",
        labels={
            "alert": "Alert Level",
            "count": "Number of Earthquakes"
        }
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(20), use_container_width=True)


with model_tab:
    st.subheader("Logistic Regression Model")

    st.write(
        "The model predicts earthquake alert level using "
        "`magnitude`, `depth`, `cdi`, `mmi`, and `sig`."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Logistic Regression")
    col2.metric("Training Rows", f"{len(X_train):,}")
    col3.metric("Test Accuracy", f"{accuracy:.2%}")

    st.markdown("### Classification Report")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown("### Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        labels={
            "x": "Predicted Alert",
            "y": "Actual Alert",
            "color": "Count"
        },
        title="Confusion Matrix"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### Logistic Regression Coefficients")

    logistic_model = model.named_steps["logistic_regression"]

    coef_df = pd.DataFrame(
        logistic_model.coef_,
        columns=feature_cols,
        index=labels
    )

    st.dataframe(coef_df, use_container_width=True)

    coef_long = (
        coef_df
        .reset_index()
        .rename(columns={"index": "alert"})
        .melt(
            id_vars="alert",
            var_name="feature",
            value_name="coefficient"
        )
    )

    fig_coef = px.bar(
        coef_long,
        x="feature",
        y="coefficient",
        color="alert",
        barmode="group",
        title="Model Coefficients by Alert Level"
    )

    st.plotly_chart(fig_coef, use_container_width=True)


with predict_tab:
    st.subheader("Predict Earthquake Alert Level")

    st.write("Enter earthquake values and the model will predict the alert level.")

    input_values = {}

    col1, col2 = st.columns(2)

    with col1:
        input_values["magnitude"] = st.slider(
            "Magnitude",
            float(df["magnitude"].min()),
            float(df["magnitude"].max()),
            float(df["magnitude"].median())
        )

        input_values["depth"] = st.slider(
            "Depth",
            float(df["depth"].min()),
            float(df["depth"].max()),
            float(df["depth"].median())
        )

        input_values["sig"] = st.slider(
            "Significance",
            float(df["sig"].min()),
            float(df["sig"].max()),
            float(df["sig"].median())
        )

    with col2:
        input_values["cdi"] = st.slider(
            "CDI",
            float(df["cdi"].min()),
            float(df["cdi"].max()),
            float(df["cdi"].median())
        )

        input_values["mmi"] = st.slider(
            "MMI",
            float(df["mmi"].min()),
            float(df["mmi"].max()),
            float(df["mmi"].median())
        )

    input_df = pd.DataFrame([input_values])[feature_cols]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.success(f"Predicted Alert Level: **{prediction.upper()}**")

    probability_df = pd.DataFrame(
        {
            "alert": model.classes_,
            "probability": probabilities
        }
    ).sort_values("probability", ascending=False)

    st.markdown("### Prediction Probabilities")
    st.dataframe(probability_df, use_container_width=True)

    fig_prob = px.bar(
        probability_df,
        x="alert",
        y="probability",
        title="Prediction Probability by Alert Level",
        text="probability"
    )

    fig_prob.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig_prob.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig_prob, use_container_width=True)