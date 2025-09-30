import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import numpy as np



# -----------------------------
# 1. Load and preprocess data
# -----------------------------
@st.cache_data
def load_data(CSV_PATH, SPEED_LIMIT_BEFORE, SPEED_LIMIT_AFTER, POLICY_CHANGE_DATE):
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df["Period"] = df["Date"].apply(lambda x: "<-Before 20 Mph" if x < POLICY_CHANGE_DATE else "->After 20 Mph")
    #df["Hour"] = df["Date"].dt.hour
    df["Weekday"] = df["Date"].dt.day_name()
    df["SpeedLimit"] = df["Period"].apply(lambda x: SPEED_LIMIT_BEFORE if x == "<-Before 20 Mph" else SPEED_LIMIT_AFTER)
    df["Speeding"] = df["Average speed"] > df["SpeedLimit"]
    df["Severity"] = (df["Average speed"] - df["SpeedLimit"]).clip(lower=0) * df["Number of vehicles"]
    return df

# -----------------------------
# 2. Sidebar controls
# -----------------------------
def add_sidebar(df):
    st.sidebar.title("Filters")
    sources = df["Source"].unique()
    selected_sources = st.sidebar.multiselect("Radar Source", sources, default=list(sources))
    #selected_period = st.sidebar.radio("Period", ["Both", "<-Before 20 Mph", "->After 20 Mph"])

    if not selected_sources:
        st.warning("Please select at least one radar source to view data.")
        st.stop()

    # --- APPLY FILTERS ---
    #if selected_period != "Both":
    #    df = df[df["Period"] == selected_period]
    df_filtered = df[df["Source"].isin(selected_sources)]
    return df_filtered



# -----------------------------
# 3. GDPR Statement
# -----------------------------
def show_GDPR_statement():
    with st.expander("🔐 Privacy Statement"):
        st.markdown("""
        This dashboard is built with privacy and transparency at its core. It complies with the principles of the UK GDPR and relevant data protection regulations.

        - **No personal data is collected or stored.** All vehicle data is anonymized and aggregated. No license plates, driver identities, or tracking information are included.
        - **TimeBlock grouping is used** to avoid exposing exact timestamps, ensuring behavioral trends are visible without compromising individual privacy.
        - **Purpose limitation:** Data is used exclusively for analytical purposes — to evaluate traffic patterns and inform policy decisions. It is not used for enforcement, profiling, or commercial activities.
        - **Data minimization:** Only the minimum necessary data is processed. Visualizations reflect group-level patterns, never individual behavior.
        - **Transparency and accountability:** This dashboard is designed to be open about how data is used. If you have questions or concerns, please contact the dashboard administrator.
        - **Security and access control:** Access to raw data is restricted. All visualizations are based on aggregated summaries to prevent re-identification.

        This dashboard reflects a commitment to ethical data use — empowering insight without compromising individual rights.
        """)


# -----------------------------
# 4. SUMMARY STATS
# -----------------------------

def summary_stats(df):
    st.markdown("---")  # horizontal rule
    st.subheader("Summary Statistics")
    summary = df.groupby(["Period", "Source"]).agg({
        "Average speed": "mean", # ["mean", "std"],
        "Maximum speed": ["mean", "max"],
        "Speeding": "mean",
        "Number of vehicles": "sum"
    }).round(2)
    summary.columns = [
        "Avg Speed",
        "Avg Top Speed per hr",
        "Max Speed Recorded",
        "Speeding Rate",
        "Vehicles"
    ]

    severity_summary = df.groupby(["Period", "Source"])["Severity"].sum().reset_index()
    summary = pd.merge(summary, severity_summary, on=["Period", "Source"])
    summary = summary.drop(columns=["Speeding Rate", "Severity"])
    summary = summary[[
        "Source",
        "Period",
        "Vehicles",
        "Avg Speed",
        "Avg Top Speed per hr"
        #"Max Speed Recorded" #,
        #"Severity"
    ]]
    #summary = summary.reset_index()
    period_order = ["<-Before 20 Mph", "->After 20 Mph"]
    summary["Period"] = pd.Categorical(summary["Period"], categories=period_order, ordered=True)
    summary = summary.sort_values(by=["Source", "Period"])

    st.dataframe(summary, use_container_width=True, hide_index=True)



# -----------------------------
# 5. Visualizations
# -----------------------------


# --- Speed category distribution ---
def speed_category_distribution(df):
    st.markdown("---")  # horizontal rule
    st.subheader("Speed Category Distribution")

    # --- Define Compliance Categories ---
    df['ComplianceCategory'] = pd.cut(
        df['Average speed'],
        bins=[0, 20, 25, 30, 100],
        labels=["Under 20mph", "20-25mph", "25-30mph", "30+mph"]
    )

    # --- Streamlit Toggle ---
    view_mode = st.radio("View Mode", ["Percentage", "Raw Counts"])

    # --- Group and Summarize ---
    compliance_summary = df.groupby(['Period', 'ComplianceCategory']).size().unstack(fill_value=0)

    # Ensure column names are strings
    compliance_summary.columns = compliance_summary.columns.astype(str)

    # Reorder Periods and Categories
    compliance_summary = compliance_summary.reindex(["<-Before 20 Mph", "->After 20 Mph"])
    category_order = ["Under 20mph", "20-25mph", "25-30mph", "30+mph"]
    compliance_summary = compliance_summary[category_order]

    # Convert to percentages if selected
    if view_mode == "Percentage":
        compliance_summary = compliance_summary.apply(lambda x: x / x.sum() * 100, axis=1)

    # --- Define Custom Colors ---
    category_colors = {
        "Under 20mph": "#2ecc71",     # Green
        "20-25mph": "#f39c12",        # Amber
        "25-30mph": "#e74c3c",        # Red
        "30+mph": "#000000"           # Black
    }

    # --- Plot Stacked Bar Chart ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(compliance_summary))

    for category in category_order:
        if category in compliance_summary.columns:
            values = compliance_summary[category].values
            ax1.bar(compliance_summary.index, values, label=category, bottom=bottom, color=category_colors[category])
            for i, value in enumerate(values):
                if value > 0:
                    label = f"{value:.1f}%" if view_mode == "Percentage" else f"{int(value)}"
                    ax1.text(i, bottom[i] + value / 2, label, ha='center', va='center', color='white', fontsize=9)
            bottom += values

    ax1.set_ylabel("Percentage of Vehicles (%)" if view_mode == "Percentage" else "Number of Vehicles")
    ax1.set_title("Speed Categories Before vs After 20mph change")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], title="Speed Category", loc="center left", bbox_to_anchor=(-0.35, 0.5), ncol=1)
    #ax1.legend(title="Speed Category", loc="center right", bbox_to_anchor=(-0.15, 0.5), ncol=1)
    st.pyplot(fig1)
    st.caption("This chart shows the split of people driving in different speed bands.")



# --- Speed Distribution ---
def speeding_distribution(df):
    st.markdown("---")  # horizontal rule
    st.subheader("Speed Distribution")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Average speed",
        hue="Period",
        bins=20,
        stat="percent",  # ← This normalizes each group to show percentages
        common_norm=False,  # ← Ensures each Period is normalized independently
        kde=False,
        ax=ax2
    )

    ax2.set_title("Speed Distribution Before vs After 20 mph Limit")
    ax2.set_xlabel("Average Speed (mph)")
    ax2.set_ylabel("Percentage of Vehicles (%)")
    st.pyplot(fig2)
    st.caption("This chart shows that more people are driving at slower speeds since the speed limit change.")

# Hourly Speed Trends
def hourly_speed_trends(df):
    st.markdown("---")  # horizontal rule
    st.subheader("Time of day Speed Trends")
    # Ensure TimeBlock is ordered
    time_order = ["Overnight", "Early Morning", "Morning Rush", "Midday", "Afternoon", "Evening Rush", "Late Evening"]
    df["TimeBlock"] = pd.Categorical(df["TimeBlock"], categories=time_order, ordered=True)
    hourly = df.groupby(["TimeBlock", "Period"])["Average speed"].mean().reset_index()
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=hourly, x="TimeBlock", y="Average speed", hue="Period", ax=ax3)
    #fig3, ax3 = plt.subplots(figsize=(12, 6))
    #sns.barplot(data=hourly, x="Hour", y="Average speed", hue="Period", ax=ax3)
    ax3.set_title("Average Speed by Time of Day")
    ax3.set_xlabel("")
    ax3.set_ylabel("Average Speed (mph)")
    ax3.set_xticks(range(0, 7))
    ax3.set_ylim(10, 40)

    st.pyplot(fig3)

# Heatmap: Time of Day vs Weekday
def plot_heatmap(df):
    st.markdown("---")  # horizontal rule
    st.subheader("Average Speed Heatmap (Time of Day vs Weekday)")

    # Period toggle
    selected_period = st.radio(
        "Select Period",
        ["<-Before 20 Mph", "->After 20 Mph"],
        index=1  # ← This makes "->After 20 Mph" the default
    )

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter by selected period
    df_filtered = df[df["Period"] == selected_period]

    # Ensure Weekday is ordered
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_type = CategoricalDtype(categories=weekday_order, ordered=True)

    # Set index and extract weekday
    df_filtered = df_filtered.set_index("Date").sort_index()
    df_filtered["Weekday"] = df_filtered.index.day_name().astype(weekday_type)

    # Ensure TimeBlock is ordered
    time_order = ["Overnight", "Early Morning", "Morning Rush", "Midday", "Afternoon", "Evening Rush", "Late Evening"]
    df_filtered["TimeBlock"] = pd.Categorical(df_filtered["TimeBlock"], categories=time_order, ordered=True)

    # Group and pivot
    heatmap_data = df_filtered.groupby(["Weekday", "TimeBlock"])["Average speed"].mean().unstack()
    heatmap_data = heatmap_data.reindex(index=weekday_order, columns=time_order)
    heatmap_data = heatmap_data.round(0).fillna(0).astype(int)

    # Plot
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt="d", ax=ax4)
    ax4.set_title(f"Average Speed by Time of Day and Weekday ({selected_period})")
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    st.pyplot(fig4)
    st.caption("This chart shows the average speed across the week and the time of day.")


# Heatmap: Delta After vs Before
def plot_delta_heatmap(df):
    st.markdown("---")  # horizontal rule
    st.subheader("Change in Average Speed since 20 mph change")

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure Weekday is ordered
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_type = CategoricalDtype(categories=weekday_order, ordered=True)

    # Set index and extract weekday
    df = df.set_index("Date").sort_index()
    df["Weekday"] = df.index.day_name().astype(weekday_type)

    # Ensure TimeBlock is ordered
    time_order = ["Overnight", "Early Morning", "Morning Rush", "Midday", "Afternoon", "Evening Rush", "Late Evening"]
    df["TimeBlock"] = pd.Categorical(df["TimeBlock"], categories=time_order, ordered=True)

    # Split by Period
    before_df = df[df["Period"] == "<-Before 20 Mph"]
    after_df = df[df["Period"] == "->After 20 Mph"]

    # Group and pivot
    before_heatmap = before_df.groupby(["Weekday", "TimeBlock"])["Average speed"].mean().unstack()
    after_heatmap = after_df.groupby(["Weekday", "TimeBlock"])["Average speed"].mean().unstack()

    # Calculate delta
    delta_heatmap = after_heatmap - before_heatmap
    delta_heatmap = delta_heatmap.reindex(weekday_order)
    delta_heatmap = delta_heatmap.round(1)

    # Plot
    fig5, ax5 = plt.subplots(figsize=(14, 6))
    sns.heatmap(delta_heatmap, cmap="RdBu_r", center=0, annot=True, fmt=".1f", ax=ax5)
    ax5.set_title("Change in Average Speed by Time of Day and Weekday (since 20 mph change)")
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    st.pyplot(fig5)
    st.caption("This chart shows the change in average speed since the speed limit change.")


# Rolling Average Speed
def rolling_average_speed(df, POLICY_CHANGE_DATE):
    st.markdown("---")  # horizontal rule
    st.subheader("7-Day Rolling Average Speed")
    
    # Ensure Weekday is ordered
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_type = CategoricalDtype(categories=weekday_order, ordered=True)
    
    df = df.set_index("Date")  # Set datetime index

    df["Weekday"] = df.index.day_name().astype(weekday_type)
    #df["Weekday"] = df["Date"].dt.day_name().astype(weekday_type)
    
    # Resample and compute rolling average
    hourly_df = df.resample("h").agg({
        "Average speed": "mean",
        "Maximum speed": "mean",
        "Number of vehicles": "sum",
        "Speeding": "mean",
        "Source": lambda x: x.mode()[0] if not x.mode().empty else None
    }).dropna()

    hourly_df["RollingAvgSpeed"] = hourly_df["Average speed"].rolling(window=24*7).mean()

    # Plot
    fig6, ax6 = plt.subplots(figsize=(18, 6))
    sns.lineplot(data=hourly_df, x=hourly_df.index, y="RollingAvgSpeed", hue="Source", ax=ax6)
    ax6.axvline(POLICY_CHANGE_DATE, color="red", linestyle="--", label="Speed Limit Change")
    ax6.set_ylim(10, 40)
    ax6.set_title("7-Day Rolling Average Speed")
    ax6.legend()
    st.pyplot(fig6)
    st.caption("This chart shows the average speed since the radar signs were installed.")


# Scatter Plot: Speed vs Time
def scatter_speed_time(df, POLICY_CHANGE_DATE):
    st.markdown("---")  # horizontal rule
    st.subheader("Scatter Plot: Average Speed over Time")
    
    df = df.set_index("Date")  # Set datetime index
    
    fig7, ax7 = plt.subplots(figsize=(18, 6))
    sns.scatterplot(data=df, x=df.index, y="Average speed", hue="Period", alpha=0.5, ax=ax7)
    ax7.axvline(POLICY_CHANGE_DATE, color="red", linestyle="--", label="Speed Limit Change")
    ax7.set_title("Average Speed Over Time")
    ax7.set_xlabel("Date")
    ax7.set_ylabel("Average Speed (mph)")
    ax7.legend()

    st.pyplot(fig7)

# -----------------------------
# 5. Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Speed Dashboard", layout="wide")
    st.title("🚦 Radar Speed Sign Dashboard")

    # --- CONFIG ---
    CSV_PATH = "Combined_20250702.csv"
    SPEED_LIMIT_BEFORE = 30
    SPEED_LIMIT_AFTER = 20
    POLICY_CHANGE_DATE = pd.to_datetime("2023-09-17")

    df = load_data(CSV_PATH, SPEED_LIMIT_BEFORE, SPEED_LIMIT_AFTER, POLICY_CHANGE_DATE)
    
    filtered_df = add_sidebar(df)
    show_GDPR_statement()

    summary_stats(filtered_df)
    speed_category_distribution(filtered_df)
    speeding_distribution(filtered_df)
    hourly_speed_trends(filtered_df)
    plot_heatmap(filtered_df)
    plot_delta_heatmap(filtered_df)
    rolling_average_speed(filtered_df, POLICY_CHANGE_DATE)
    scatter_speed_time(filtered_df, POLICY_CHANGE_DATE)

if __name__ == "__main__":
    main()