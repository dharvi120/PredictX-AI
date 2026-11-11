
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

REPORT_FILE = (
    ROOT / "reports" / "predictx_maintenance_decisions.csv"
)

st.set_page_config(
    page_title="PredictX AI | Command Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design system
BG = "#080F1B"
PANEL = "#111D2D"
PANEL_2 = "#162438"
BORDER = "#24364B"
TEXT = "#E8F0FA"
MUTED = "#8FA4BB"
CYAN = "#38D9F5"
BLUE = "#438BFF"
GREEN = "#36D399"
AMBER = "#F6B94A"
RED = "#FF657A"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 0%,
        rgba(33, 100, 160, 0.13), transparent 30%),
        #080F1B;
    color: #E8F0FA;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #0D1725;
    border-right: 1px solid #24364B;
}

[data-testid="stSidebar"] * {
    color: #DCE8F5;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #F3F7FC;
}

p, label, span {
    color: inherit;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #162438, #101B2A);
    border: 1px solid #2A3D54;
    border-radius: 13px;
    padding: 18px 20px;
}

[data-testid="stMetricLabel"] {
    color: #94A9C1 !important;
    font-size: 12px !important;
}

[data-testid="stMetricValue"] {
    color: #F4F8FF !important;
    font-family: 'Space Grotesk', sans-serif;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #162438;
    border-color: #31465E;
    color: white;
}

.stButton button {
    border-radius: 9px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #24364B;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #24364B;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PLOTLY THEME
# =========================================================

def style_chart(fig, height=340):

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans",
            color=TEXT,
            size=12
        ),
        title_font=dict(
            family="Space Grotesk",
            size=16,
            color=TEXT
        ),
        margin=dict(l=20, r=20, t=55, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED)
        ),
        xaxis=dict(
            gridcolor="#24364B",
            zerolinecolor="#24364B",
            color=MUTED
        ),
        yaxis=dict(
            gridcolor="#24364B",
            zerolinecolor="#24364B",
            color=MUTED
        )
    )

    return fig


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(path, modified_time):

    df = pd.read_csv(path)

    required = [
        "unit_id",
        "predicted_rul",
        "lower_rul",
        "upper_rul",
        "interval_width",
        "average_sensor_health",
        "maintenance_priority",
        "sensor_health_status",
        "prediction_trust",
        "recommended_action"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


if not REPORT_FILE.exists():

    st.error(
        "Maintenance report not found. "
        "Run src/build_decision_report.py first."
    )
    st.stop()

try:

    df = load_data(
        REPORT_FILE,
        REPORT_FILE.stat().st_mtime
    )

except Exception as error:

    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:12px 0 24px 0;">
            <div style="font-size:25px;font-weight:800;
                        color:#F4F8FF;font-family:Space Grotesk;">
                ⚙ PredictX<span style="color:#38D9F5;"> AI</span>
            </div>
            <div style="font-size:10px;color:#8FA4BB;
                        letter-spacing:2px;margin-top:5px;">
                INTELLIGENT ASSET MONITORING
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Fleet Intelligence",
            "Engine Inspection"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### DATA SOURCE")

    st.markdown(
        """
        <div style="background:#162438;border:1px solid #2A3D54;
                    border-radius:10px;padding:13px;">
            <div style="font-size:11px;color:#8FA4BB;">
                ACTIVE DATASET
            </div>
            <div style="font-weight:700;color:#F3F7FC;
                        margin-top:5px;">
                NASA C-MAPSS FD001
            </div>
            <div style="font-size:11px;color:#36D399;
                        margin-top:8px;">
                ● Report loaded
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("PredictX AI • Research Prototype")
    st.caption("Reliability-aware predictive maintenance")


# =========================================================
# SHARED METRICS
# =========================================================

total = len(df)

high = int(
    (df["maintenance_priority"] == "High").sum()
)

medium = int(
    (df["maintenance_priority"] == "Medium").sum()
)

low = int(
    (df["maintenance_priority"] == "Low").sum()
)

avg_rul = df["predicted_rul"].mean()
avg_health = df["average_sensor_health"].mean()
avg_width = df["interval_width"].mean()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div style="display:flex;justify-content:space-between;
                align-items:center;flex-wrap:wrap;gap:12px;">
        <div>
            <div style="font-size:11px;color:#38D9F5;
                        letter-spacing:2px;font-weight:700;">
                PREDICTIVE INTELLIGENCE PLATFORM
            </div>
            <div style="font-family:Space Grotesk;
                        font-size:30px;font-weight:700;
                        color:#F3F7FC;margin-top:5px;">
                PredictX AI Command Center
            </div>
            <div style="color:#8FA4BB;font-size:13px;
                        margin-top:5px;">
                Monitor asset health, estimate remaining useful life,
                and assess prediction reliability.
            </div>
        </div>
        <div style="background:#122A2B;border:1px solid #28645B;
                    border-radius:30px;padding:9px 15px;
                    color:#36D399;font-size:12px;">
            ● REPORT AVAILABLE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# COMMAND CENTER
# =========================================================

if page == "Command Center":

    st.markdown("## System Overview")

    cols = st.columns(5)

    metrics = [
        ("MONITORED ENGINES", f"{total}", "Asset fleet"),
        ("HIGH PRIORITY", f"{high}", "Requires attention"),
        ("AVERAGE RUL", f"{avg_rul:.1f}", "Operating cycles"),
        ("SENSOR HEALTH", f"{avg_health:.1f}", "Out of 100"),
        ("INTERVAL WIDTH", f"{avg_width:.1f}", "Average cycles")
    ]

    for col, (label, value, caption) in zip(cols, metrics):

        with col:

            st.markdown(
                f"""
                <div style="background:linear-gradient(145deg,
                    #17283D,#101B2A);
                    border:1px solid #2B4058;
                    border-radius:13px;
                    padding:17px;height:112px;">

                    <div style="font-size:10px;color:#8FA4BB;
                                letter-spacing:1px;">
                        {label}
                    </div>

                    <div style="font-family:Space Grotesk;
                                font-size:27px;font-weight:700;
                                color:#F4F8FF;margin-top:10px;">
                        {value}
                    </div>

                    <div style="font-size:10px;color:#8FA4BB;
                                margin-top:3px;">
                        {caption}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # Priority and sensor health

    st.markdown("## Fleet Risk Intelligence")

    left, right = st.columns(2, gap="large")

    with left:

        priority_data = pd.DataFrame({
            "Priority": ["High", "Medium", "Low"],
            "Engines": [high, medium, low]
        })

        fig = px.bar(
            priority_data,
            x="Priority",
            y="Engines",
            color="Priority",
            color_discrete_map={
                "High": RED,
                "Medium": AMBER,
                "Low": GREEN
            },
            title="Maintenance Priority Distribution"
        )

        fig.update_traces(
            marker_line_width=0,
            width=0.48
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:

        health_data = (
            df["sensor_health_status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Engines")
        )

        fig = px.pie(
            health_data,
            names="Status",
            values="Engines",
            hole=0.68,
            color="Status",
            color_discrete_map={
                "Healthy": GREEN,
                "Warning": AMBER,
                "Critical": RED
            },
            title="Sensor Health Distribution"
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.divider()

    # RUL and uncertainty

    st.markdown("## Remaining Useful Life Intelligence")

    left, right = st.columns(2, gap="large")

    with left:

        fig = px.histogram(
            df,
            x="predicted_rul",
            nbins=24,
            title="Predicted RUL Distribution",
            color_discrete_sequence=[CYAN]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with right:

        fig = px.scatter(
            df,
            x="predicted_rul",
            y="interval_width",
            color="maintenance_priority",
            hover_data=["unit_id"],
            color_discrete_map={
                "High": RED,
                "Medium": AMBER,
                "Low": GREEN
            },
            title="RUL vs Prediction Uncertainty"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    # Priority register

    st.markdown("## Priority Attention Register")

    attention = (
        df[df["maintenance_priority"] == "High"]
        .sort_values("predicted_rul")
    )

    st.dataframe(
        attention[
            [
                "unit_id",
                "predicted_rul",
                "average_sensor_health",
                "interval_width",
                "recommended_action"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FLEET INTELLIGENCE
# =========================================================

elif page == "Fleet Intelligence":

    st.markdown("## Fleet Intelligence")

    st.caption(
        "Explore engine-level predictions and maintenance priorities."
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_priority = st.multiselect(
            "Maintenance priority",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )

    with col2:

        min_health = st.slider(
            "Minimum sensor health",
            0,
            100,
            0
        )

    filtered = df[
        df["maintenance_priority"].isin(selected_priority)
        & (df["average_sensor_health"] >= min_health)
    ]

    st.metric("Engines matching filters", len(filtered))

    st.dataframe(
        filtered.sort_values(
            "predicted_rul",
            ascending=True
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download filtered fleet report",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="predictx_fleet_report.csv",
        mime="text/csv"
    )


# =========================================================
# ENGINE INSPECTION
# =========================================================

elif page == "Engine Inspection":

    st.markdown("## Engine Inspection Workspace")

    selected_engine = st.selectbox(
        "Choose an engine",
        sorted(df["unit_id"].unique())
    )

    engine = df[
        df["unit_id"] == selected_engine
    ].iloc[0]

    st.markdown(
        f"### Engine #{int(selected_engine)}"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Predicted RUL",
        f"{engine['predicted_rul']:.1f} cycles"
    )

    b.metric(
        "Lower Bound",
        f"{engine['lower_rul']:.1f} cycles"
    )

    c.metric(
        "Upper Bound",
        f"{engine['upper_rul']:.1f} cycles"
    )

    d.metric(
        "Sensor Health",
        f"{engine['average_sensor_health']:.1f}/100"
    )

    st.write("")

    left, right = st.columns([1.5, 1])

    with left:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=["Engine", "Engine"],
                y=[
                    engine["lower_rul"],
                    engine["upper_rul"]
                ],
                mode="lines",
                name="Prediction interval",
                line=dict(color=CYAN, width=12)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=["Engine"],
                y=[engine["predicted_rul"]],
                mode="markers",
                name="Predicted RUL",
                marker=dict(
                    color=AMBER,
                    size=15,
                    line=dict(color="white", width=1)
                )
            )
        )

        fig.update_layout(
            title="Remaining Useful Life Estimate",
            yaxis_title="RUL (cycles)",
            xaxis=dict(showticklabels=False)
        )

        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True
        )

    with right:

        st.markdown("### Decision Summary")

        st.markdown(
            f"""
            <div style="background:#14243A;
                        border:1px solid #2B4058;
                        border-radius:13px;padding:18px;">

                <div style="color:#8FA4BB;font-size:11px;">
                    MAINTENANCE PRIORITY
                </div>

                <div style="font-size:24px;font-weight:700;
                            margin-top:6px;">
                    {engine['maintenance_priority']}
                </div>

                <hr style="border-color:#2B4058;">

                <div style="color:#8FA4BB;font-size:11px;">
                    SENSOR HEALTH STATUS
                </div>

                <div style="font-size:19px;font-weight:600;
                            margin-top:5px;">
                    {engine['sensor_health_status']}
                </div>

                <hr style="border-color:#2B4058;">

                <div style="color:#8FA4BB;font-size:11px;">
                    PREDICTION TRUST
                </div>

                <div style="font-size:17px;font-weight:600;
                            margin-top:5px;">
                    {engine['prediction_trust']}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Recommended Action")

    st.info(engine["recommended_action"])

    st.caption(
        "The displayed maintenance priority is generated by "
        "experimental prototype rules, not a validated industrial policy."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="display:flex;justify-content:space-between;
                flex-wrap:wrap;gap:10px;color:#71869E;
                font-size:11px;">
        <span>PredictX AI • Reliability-Aware Predictive Maintenance</span>
        <span>NASA C-MAPSS FD001 • Experimental Prototype</span>
    </div>
    """,
    unsafe_allow_html=True
)
