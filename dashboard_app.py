import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hyundai Delay Reporting Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = Path("data/VIN_XGBoost_Clean_Model(5).xlsx")
MAIN_SHEET = "Unified VIN Risk Scoring"
ML_IMPORTANCE_SHEET = "ML Feature Importance"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .subtitle {
            color: #667085;
            font-size: 0.98rem;
            margin-bottom: 1.25rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 750;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #eaecf0;
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_workbook(uploaded_file=None):
    if uploaded_file is not None:
        vin_df = pd.read_excel(uploaded_file, sheet_name=MAIN_SHEET)
        try:
            ml_df = pd.read_excel(uploaded_file, sheet_name=ML_IMPORTANCE_SHEET)
        except Exception:
            ml_df = pd.DataFrame()
        return vin_df, ml_df

    if DATA_PATH.exists():
        vin_df = pd.read_excel(DATA_PATH, sheet_name=MAIN_SHEET)
        try:
            ml_df = pd.read_excel(DATA_PATH, sheet_name=ML_IMPORTANCE_SHEET)
        except Exception:
            ml_df = pd.DataFrame()
        return vin_df, ml_df

    return pd.DataFrame(), pd.DataFrame()


def clean_vin_data(df):
    df = df.copy()

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Required numeric fields
    numeric_cols = [
        "Days Down",
        "Repeat Repairs",
        "Dealer NPS",
        "Dealer Repeat Repair %",
        "Dealer Training Completion %",
        "Agent Transfers",
        "Missed Callbacks",
        "CAC Contacts",
        "Case Reopens",
        "Total Risk Score",
        "Capped Risk Score",
        "Idiosyncratic Risk",
        "Systemic Risk",
        "Time Decay",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Text fields
    text_cols = [
        "VIN",
        "Dealer Name",
        "Area Manager",
        "Mobility Status",
        "Risk Band",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Aging bucket
    df["Aging Bucket"] = pd.cut(
        df["Days Down"],
        bins=[-1, 7, 14, 21, 30, 45, 10_000],
        labels=["0–7 Days", "8–14 Days", "15–21 Days", "22–30 Days", "31–45 Days", "46+ Days"],
    )

    # Standardize critical/severe flags
    df["Is Critical"] = df["Risk Band"].str.contains("Critical", case=False, na=False)
    df["Is High Risk"] = df["Risk Band"].str.contains("High", case=False, na=False)
    df["Is Severe Or Critical"] = df["Is Critical"] | df["Is High Risk"]

    return df


def build_driver_summary(df, ml_df):
    """
    Creates a management-level view of which drivers are impacting the active VIN pool.
    Combines frequency/severity with ML feature importance where available.
    """

    driver_rules = [
        {
            "Driver": "Prolonged Downtime",
            "Column": "Days Down",
            "Condition": df["Days Down"] >= 30,
            "Avg Value": df.loc[df["Days Down"] >= 30, "Days Down"].mean(),
        },
        {
            "Driver": "Repeat Repairs",
            "Column": "Repeat Repairs",
            "Condition": df["Repeat Repairs"] >= 3,
            "Avg Value": df.loc[df["Repeat Repairs"] >= 3, "Repeat Repairs"].mean(),
        },
        {
            "Driver": "No Loaner / Mobility Gap",
            "Column": "Mobility Status",
            "Condition": df["Mobility Status"].str.contains("No Loaner", case=False, na=False),
            "Avg Value": None,
        },
        {
            "Driver": "Low Dealer NPS",
            "Column": "Dealer NPS",
            "Condition": df["Dealer NPS"] < 75,
            "Avg Value": df.loc[df["Dealer NPS"] < 75, "Dealer NPS"].mean(),
        },
        {
            "Driver": "High Dealer Repeat Repair %",
            "Column": "Dealer Repeat Repair %",
            "Condition": df["Dealer Repeat Repair %"] >= 9,
            "Avg Value": df.loc[df["Dealer Repeat Repair %"] >= 9, "Dealer Repeat Repair %"].mean(),
        },
        {
            "Driver": "Low Training Completion",
            "Column": "Dealer Training Completion %",
            "Condition": df["Dealer Training Completion %"] < 70,
            "Avg Value": df.loc[df["Dealer Training Completion %"] < 70, "Dealer Training Completion %"].mean(),
        },
        {
            "Driver": "High Agent Transfers",
            "Column": "Agent Transfers",
            "Condition": df["Agent Transfers"] >= 5,
            "Avg Value": df.loc[df["Agent Transfers"] >= 5, "Agent Transfers"].mean(),
        },
        {
            "Driver": "Missed Callbacks",
            "Column": "Missed Callbacks",
            "Condition": df["Missed Callbacks"] >= 3,
            "Avg Value": df.loc[df["Missed Callbacks"] >= 3, "Missed Callbacks"].mean(),
        },
        {
            "Driver": "High CAC Contacts",
            "Column": "CAC Contacts",
            "Condition": df["CAC Contacts"] >= 5,
            "Avg Value": df.loc[df["CAC Contacts"] >= 5, "CAC Contacts"].mean(),
        },
        {
            "Driver": "Case Reopens",
            "Column": "Case Reopens",
            "Condition": df["Case Reopens"] >= 2,
            "Avg Value": df.loc[df["Case Reopens"] >= 2, "Case Reopens"].mean(),
        },
    ]

    rows = []

    for rule in driver_rules:
        condition = rule["Condition"]
        impacted = df[condition]

        rows.append(
            {
                "Driver": rule["Driver"],
                "Impacted VINs": len(impacted),
                "Critical VINs": int(impacted["Is Critical"].sum()) if not impacted.empty else 0,
                "Avg Risk Score": impacted["Capped Risk Score"].mean() if not impacted.empty else 0,
                "Avg Days Down": impacted["Days Down"].mean() if not impacted.empty else 0,
                "Avg Driver Value": rule["Avg Value"] if rule["Avg Value"] is not None else "",
            }
        )

    driver_df = pd.DataFrame(rows)

    # Add ML importance where available
    if not ml_df.empty and {"Feature", "Importance"}.issubset(set(ml_df.columns)):
        importance_map = {
            "Days Down": "Days Down",
            "Repeat Repairs": "Repeat Repairs",
            "No Loaner / Mobility Gap": "Mobility Status",
            "Low Dealer NPS": "Dealer NPS",
            "High Dealer Repeat Repair %": "Dealer Repeat Repair %",
            "Low Training Completion": "Dealer Training Completion %",
            "High Agent Transfers": "Agent Transfers",
            "Missed Callbacks": "Missed Callbacks",
            "High CAC Contacts": "CAC Contacts",
            "Case Reopens": "Case Reopens",
        }

        ml_lookup = ml_df.set_index("Feature")["Importance"].to_dict()
        driver_df["ML Importance"] = driver_df["Driver"].map(
            lambda x: ml_lookup.get(importance_map.get(x, ""), 0)
        )
    else:
        driver_df["ML Importance"] = 0

    driver_df = driver_df.sort_values(
        by=["Impacted VINs", "Critical VINs", "ML Importance", "Avg Risk Score"],
        ascending=[False, False, False, False],
    )

    return driver_df


# ============================================================
# HEADER
# ============================================================

st.title("Hyundai Delay Reporting Dashboard")

st.markdown(
    """
    <div class="subtitle">
    Manager-level portfolio view of active delayed vehicles, aging repair exposure,
    ongoing risk concentration, highest impacted dealers, and the operational drivers
    creating the greatest escalation risk.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA SOURCE
# ============================================================

with st.sidebar:
    st.header("Data Source")

    uploaded_file = st.file_uploader(
        "Upload model workbook",
        type=["xlsx", "xls"],
        help="Upload the VIN_XGBoost_Clean_Model workbook.",
    )

    st.caption("If no file is uploaded, the app uses `data/VIN_XGBoost_Clean_Model(5).xlsx`.")


raw_df, ml_importance_df = load_workbook(uploaded_file)

if raw_df.empty:
    st.warning("No workbook found. Upload the Excel file or place it inside the `data/` folder.")
    st.stop()

df = clean_vin_data(raw_df)


# ============================================================
# FILTERS
# ============================================================

with st.sidebar:
    st.header("Filters")

    area_manager_filter = st.multiselect(
        "Area Manager",
        options=sorted(df["Area Manager"].dropna().unique()),
        default=sorted(df["Area Manager"].dropna().unique()),
    )

    dealer_filter = st.multiselect(
        "Dealer",
        options=sorted(df["Dealer Name"].dropna().unique()),
        default=sorted(df["Dealer Name"].dropna().unique()),
    )

    risk_band_filter = st.multiselect(
        "Risk Band",
        options=sorted(df["Risk Band"].dropna().unique()),
        default=sorted(df["Risk Band"].dropna().unique()),
    )

    mobility_filter = st.multiselect(
        "Mobility Status",
        options=sorted(df["Mobility Status"].dropna().unique()),
        default=sorted(df["Mobility Status"].dropna().unique()),
    )


filtered_df = df[
    df["Area Manager"].isin(area_manager_filter)
    & df["Dealer Name"].isin(dealer_filter)
    & df["Risk Band"].isin(risk_band_filter)
    & df["Mobility Status"].isin(mobility_filter)
].copy()

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

total_vins = len(filtered_df)
critical_vins = int(filtered_df["Is Critical"].sum())
high_risk_vins = int(filtered_df["Is High Risk"].sum())
avg_days_down = filtered_df["Days Down"].mean()
avg_risk_score = filtered_df["Capped Risk Score"].mean()
dealers_with_critical = filtered_df.loc[filtered_df["Is Critical"], "Dealer Name"].nunique()

summary_cols = st.columns(6)

summary_cols[0].metric("Total Active VINs", f"{total_vins:,}")
summary_cols[1].metric("Critical VINs", f"{critical_vins:,}")
summary_cols[2].metric("High Escalation Risk", f"{high_risk_vins:,}")
summary_cols[3].metric("Avg Days Down", f"{avg_days_down:.1f}")
summary_cols[4].metric("Avg Risk Score", f"{avg_risk_score:.2f}")
summary_cols[5].metric("Dealers w/ Critical", f"{dealers_with_critical:,}")


# ============================================================
# HISTOGRAMS
# ============================================================

st.markdown('<div class="section-title">Portfolio Risk Distribution</div>', unsafe_allow_html=True)

chart_col_1, chart_col_2 = st.columns(2)

aging_order = ["0–7 Days", "8–14 Days", "15–21 Days", "22–30 Days", "31–45 Days", "46+ Days"]

aging_counts = (
    filtered_df["Aging Bucket"]
    .value_counts()
    .reindex(aging_order)
    .fillna(0)
    .reset_index()
)

aging_counts.columns = ["Aging Bucket", "Vehicle Count"]

fig_aging = px.bar(
    aging_counts,
    x="Aging Bucket",
    y="Vehicle Count",
    text="Vehicle Count",
    title="Aging Repairs by Days Down",
)

fig_aging.update_layout(
    height=420,
    xaxis_title="Days Down Bucket",
    yaxis_title="Vehicles",
    title_x=0.02,
)

chart_col_1.plotly_chart(fig_aging, use_container_width=True)


risk_counts = (
    filtered_df["Risk Band"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = ["Risk Band", "Vehicle Count"]

fig_risk = px.bar(
    risk_counts,
    x="Risk Band",
    y="Vehicle Count",
    text="Vehicle Count",
    title="Ongoing Risk Distribution",
)

fig_risk.update_layout(
    height=420,
    xaxis_title="Risk Band",
    yaxis_title="Vehicles",
    title_x=0.02,
)

chart_col_2.plotly_chart(fig_risk, use_container_width=True)


# ============================================================
# HIGHEST IMPACTED DEALERS
# ============================================================

st.markdown('<div class="section-title">Highest Impacted Dealers</div>', unsafe_allow_html=True)

dealer_summary = (
    filtered_df.groupby("Dealer Name")
    .agg(
        Active_VINs=("VIN", "count"),
        Critical_VINs=("Is Critical", "sum"),
        High_Risk_VINs=("Is High Risk", "sum"),
        Avg_Days_Down=("Days Down", "mean"),
        Avg_Risk_Score=("Capped Risk Score", "mean"),
        Avg_Repeat_Repairs=("Repeat Repairs", "mean"),
        Avg_Dealer_NPS=("Dealer NPS", "mean"),
        Avg_Dealer_RR=("Dealer Repeat Repair %", "mean"),
        Avg_Training=("Dealer Training Completion %", "mean"),
    )
    .reset_index()
)

dealer_summary = dealer_summary.sort_values(
    by=["Critical_VINs", "High_Risk_VINs", "Active_VINs", "Avg_Risk_Score"],
    ascending=[False, False, False, False],
)

dealer_display = dealer_summary.copy()
round_cols = [
    "Avg_Days_Down",
    "Avg_Risk_Score",
    "Avg_Repeat_Repairs",
    "Avg_Dealer_NPS",
    "Avg_Dealer_RR",
    "Avg_Training",
]

for col in round_cols:
    dealer_display[col] = dealer_display[col].round(2)

dealer_col_1, dealer_col_2 = st.columns([1.25, 1])

with dealer_col_1:
    st.dataframe(
        dealer_display.rename(
            columns={
                "Dealer Name": "Dealer",
                "Active_VINs": "Active VINs",
                "Critical_VINs": "Critical VINs",
                "High_Risk_VINs": "High Risk VINs",
                "Avg_Days_Down": "Avg Days Down",
                "Avg_Risk_Score": "Avg Risk Score",
                "Avg_Repeat_Repairs": "Avg Repeat Repairs",
                "Avg_Dealer_NPS": "Avg Dealer NPS",
                "Avg_Dealer_RR": "Avg Dealer RR %",
                "Avg_Training": "Avg Training %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with dealer_col_2:
    top_dealers = dealer_summary.head(10)

    fig_dealers = px.bar(
        top_dealers,
        x="Active_VINs",
        y="Dealer Name",
        orientation="h",
        text="Active_VINs",
        title="Top Dealers by Active VIN Count",
    )

    fig_dealers.update_layout(
        height=460,
        xaxis_title="Active VINs",
        yaxis_title="Dealer",
        yaxis={"categoryorder": "total ascending"},
        title_x=0.02,
    )

    st.plotly_chart(fig_dealers, use_container_width=True)


# ============================================================
# HIGHEST IMPACTING DRIVERS
# ============================================================

st.markdown('<div class="section-title">Highest Impacting Drivers</div>', unsafe_allow_html=True)

driver_summary = build_driver_summary(filtered_df, ml_importance_df)

driver_display = driver_summary.copy()
for col in ["Avg Risk Score", "Avg Days Down", "ML Importance"]:
    driver_display[col] = pd.to_numeric(driver_display[col], errors="coerce").fillna(0).round(3)

driver_col_1, driver_col_2 = st.columns([1.1, 1])

with driver_col_1:
    st.dataframe(
        driver_display,
        use_container_width=True,
        hide_index=True,
    )

with driver_col_2:
    top_drivers = driver_summary.head(10)

    fig_drivers = px.bar(
        top_drivers,
        x="Impacted VINs",
        y="Driver",
        orientation="h",
        text="Impacted VINs",
        title="Top Drivers by Impacted VIN Count",
    )

    fig_drivers.update_layout(
        height=460,
        xaxis_title="Impacted VINs",
        yaxis_title="Driver",
        yaxis={"categoryorder": "total ascending"},
        title_x=0.02,
    )

    st.plotly_chart(fig_drivers, use_container_width=True)


# ============================================================
# ML FEATURE IMPORTANCE
# ============================================================

if not ml_importance_df.empty and {"Feature", "Importance"}.issubset(set(ml_importance_df.columns)):
    st.markdown('<div class="section-title">ML Feature Importance</div>', unsafe_allow_html=True)

    ml_plot_df = ml_importance_df.copy()
    ml_plot_df["Importance"] = pd.to_numeric(ml_plot_df["Importance"], errors="coerce").fillna(0)
    ml_plot_df = ml_plot_df.sort_values("Importance", ascending=True)

    fig_ml = px.bar(
        ml_plot_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=ml_plot_df["Importance"].round(3),
        title="XGBoost Feature Importance",
    )

    fig_ml.update_layout(
        height=460,
        xaxis_title="Importance",
        yaxis_title="Feature",
        title_x=0.02,
    )

    st.plotly_chart(fig_ml, use_container_width=True)


# ============================================================
# VEHICLE DETAIL TABLE
# ============================================================

st.markdown('<div class="section-title">Active Vehicle Detail</div>', unsafe_allow_html=True)

detail_cols = [
    "VIN",
    "Dealer Name",
    "Area Manager",
    "Days Down",
    "Repeat Repairs",
    "Mobility Status",
    "Dealer NPS",
    "Dealer Repeat Repair %",
    "Dealer Training Completion %",
    "Capped Risk Score",
    "Risk Band",
]

detail_df = filtered_df[detail_cols].sort_values(
    by=["Capped Risk Score", "Days Down"],
    ascending=[False, False],
)

st.dataframe(
    detail_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DOWNLOAD
# ============================================================

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Dashboard Data",
    data=csv,
    file_name="filtered_hyundai_delay_dashboard.csv",
    mime="text/csv",
)