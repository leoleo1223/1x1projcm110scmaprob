"""
PredictEd: A Predictive Software for Student Performance
=======================================
A Streamlit web application that explores the relationship between
socioeconomic backgrounds, study behaviors, and academic performance.

Key Features:
- Exploratory Data Analysis (Univariate & Bivariate)
- Interactive Plotly visualizations (Sunburst, Radar, Violin)
- Machine Learning Predictive Modeling (Random Forest)

Dependencies:
streamlit, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION & STYLING
# ==========================================

# Inject custom CSS to style the Streamlit app layout, colors, and navigation bar.
CUSTOM_CSS = """
<style>
    :root {
        --bg-app: #EAF5F0; --bg-surface: #FFFFFF; --bg-elevated: #F1F8F4;
        --border-soft: #C9DDD3; --text-primary: #14352C; --text-muted: #4F6B62;
        --accent: #2E7D6B; --accent-strong: #5FE0BE; --accent-deep: #0E5C4A;
        --nav-bg: #0E2A23; --nav-text: #B6D8CC; --nav-text-active: #5FE0BE; --nav-border: #1B4D40;
    }
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: var(--text-primary); }
    .stApp { background-color: var(--bg-app); }
    .stApp a { color: var(--accent-deep); }
    hr { border-color: var(--border-soft) !important; margin: 6px 0 14px 0 !important; }

    /* Hide default Streamlit headers and footers for a cleaner look */
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > section, [data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; margin-top: 0 !important; }
    [data-testid="stMainBlockContainer"], .block-container { padding-top: 0 !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; max-width: 100% !important; }

    /* Custom Navigation Bar styling */
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) { 
        position: sticky; top: 0; z-index: 999; background-color: var(--nav-bg); 
        padding: 0px 20px !important; border-bottom: 1px solid var(--nav-border); 
        border-radius: 8px; gap: 16px !important; align-items: center !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) > div[data-testid="column"] > div,
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) .element-container { padding-top: 0 !important; padding-bottom: 0 !important; margin-top: 0 !important; margin-bottom: 0 !important; }
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stColumn"] { width: auto !important; flex: 0 0 auto !important; }
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stColumn"]:last-child { flex: 1 1 auto !important; }
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stImage"], div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stImage"] > div { margin: 0 !important; padding: 0 !important; display: flex; align-items: center; justify-content: center; transform: translateY(-4px); }
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) img { margin: 0 !important; padding: 0 !important; }
    .nav-logo { display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 50%; background-color: var(--accent-strong); color: var(--nav-bg); font-weight: 800; font-size: 0.75rem; letter-spacing: 1px; flex-shrink: 0; }

    /* Styling the Radio buttons to look like nav links */
    [data-testid="stRadio"] { margin: 0 !important; padding: 0 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] { display: flex; justify-content: flex-start; align-items: center; gap: 28px; padding: 0 !important; min-height: 0 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label svg, [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label { background: transparent !important; padding: 0 !important; margin: 0 !important; min-height: 0 !important; position: relative; }
    [data-testid="stRadio"] div[role="radiogroup"] label p { font-size: 0.95rem; color: var(--nav-text); margin: 0; padding: 4px 0; transition: color 0.2s ease; }
    [data-testid="stRadio"] div[role="radiogroup"] label p::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; background-color: var(--nav-text-active); transform: scaleX(0); transform-origin: center; transition: transform 0.2s ease; }
    [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p, [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: var(--nav-text-active); font-weight: 600; }
    [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p::after, [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p::after { transform: scaleX(1); }
    [data-testid="stRadio"] div[role="radiogroup"] label:hover p { color: var(--nav-text-active); cursor: pointer; }

    /* Buttons and Inputs */
    .stButton > button, .stDownloadButton > button { background-color: var(--accent-deep) !important; color: #FFFFFF !important; border: 1px solid var(--accent) !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
    .stButton > button *, .stDownloadButton > button * { color: #FFFFFF !important; }
    .stButton > button:hover, .stDownloadButton > button:hover { background-color: var(--accent) !important; border-color: var(--accent-strong) !important; color: #FFFFFF !important; }
    [data-baseweb="select"] > div, .stTextInput input, .stMultiSelect [data-baseweb="select"] > div { background-color: var(--bg-surface) !important; border-color: var(--border-soft) !important; color: var(--text-primary) !important; }
    [data-testid="stDataFrame"] { background-color: var(--bg-surface); border: 1px solid var(--border-soft); border-radius: 8px; }
    .stAlert { background-color: var(--bg-elevated) !important; border: 1px solid var(--border-soft) !important; color: var(--text-primary) !important; }
</style>
"""

# Apply basic page configurations
st.set_page_config(page_title="PredictEd: A Predictive Software for Student Performance", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
sns.set_theme(style="whitegrid", palette="crest")

# Standardized order for ordinal categorical data
EDU_ORDER = ["No Formal Education", "High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD"]


# ==========================================
# DATA & MODEL CACHING (HELPER FUNCTIONS)
# ==========================================

@st.cache_data(show_spinner="Loading dataset...")
def load_student_data():
    """
    Loads and initially cleans the main dataset from the local file system.
    Cached via Streamlit to prevent reloading on every UI interaction.

    Returns:
        pd.DataFrame: A pandas dataframe containing the student data.
    """
    file_path = Path(__file__).parent / "Dataset" / "Student_Performance.csv"
    df = pd.read_csv(file_path)
    # Strip any accidental trailing spaces from column names
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource(show_spinner="Training Tuned Random Forest Classifier (One Time)...")
def train_model(df_clean):
    """
    Trains a Random Forest Machine Learning model using demographic and behavioral data
    to predict the exact final grade category. Cached via st.cache_resource so the heavy
    computation only happens once per session.

    Args:
        df_clean (pd.DataFrame): The sanitized dataframe containing features and target variable.

    Returns:
        clf: The trained RandomForestClassifier model.
        acc (float): The accuracy score of the model on test data.
        X.columns (Index): The ordered list of all final encoded features expected by the model.
        X_cat.columns (Index): The list of one-hot encoded categorical features.
        features_cat (list): The original raw categorical column names.
    """
    # Separate features into Categorical and Numerical categories
    features_cat = [f for f in
                    ['Gender', 'School Type', 'Parent Education', 'Internet Access', 'Travel Time', 'Extra Activities',
                     'Study Method'] if f in df_clean.columns]
    features_num = [f for f in ['Study Hours', 'Attendance Percentage'] if f in df_clean.columns]

    # Drop rows missing crucial target or feature data
    ml_df = df_clean.dropna(subset=['Final Grade'] + features_cat + features_num).copy()

    if ml_df.empty:
        return None, None, None, None, None

    # One-Hot Encode categorical variables and merge back with numericals
    X_cat = pd.get_dummies(ml_df[features_cat], drop_first=True)
    X_num = ml_df[features_num]
    X = pd.concat([X_cat, X_num], axis=1)
    y = ml_df['Final Grade']

    # 80/20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the classifier with hyperparameter optimizations
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',  # Handles imbalanced grade distributions naturally
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc, X.columns, X_cat.columns, features_cat


# ==========================================
# APP NAVIGATION & ROUTING
# ==========================================

# Initialize session state for page routing
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"


def go_to_findings():
    st.session_state.current_page = "Findings"


menu_options = ["Home", "Findings", "Resources", "About Us"]
logo_col, menu_col, _spacer = st.columns([1, 10, 1], vertical_alignment="center")

# Render custom top navigation bar
with logo_col:
    st.markdown('<span class="nav-anchor"></span>', unsafe_allow_html=True)
    logo_path = Path(__file__).parent / "Images" / "Logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=60)
    else:
        st.markdown('<div class="nav-logo">LOGO</div>', unsafe_allow_html=True)

with menu_col:
    st.radio("Navigate:", menu_options, key="current_page", horizontal=True, label_visibility="collapsed")

# ==========================================
# PAGE 1: HOME
# ==========================================
if st.session_state.current_page == "Home":
    st.title("PredictEd: A Predictive Software for Student Performance")
    home_col1, home_col2 = st.columns([1.2, 1], gap="large")

    with home_col1:
        st.write("### Project Introduction")
        st.write(
            "Welcome to the Student Performance Analytics Dashboard. This application explores a vast dataset to analyze **the distinct relationship between a student's Socioeconomic Background, Study Behaviors, and their overall Academic Performance**.")
        st.write(
            "By looking at indicators such as Parental Education, Internet Access, Attendance, and Extracurricular Activities, our goal is to highlight educational disparities and build predictive models to better understand how background and behavior combined impact student success.")
        st.write("### What You Can Do")
        st.markdown(
            "* 📊 **Analyze Socioeconomic Trends:** Explore the distributions of student backgrounds.\n* 📈 **Visualize Performance Disparities:** Compare how different groups score on tests.\n* 🤖 **Predict Academic Outcomes:** Use machine learning to forecast grades based on demographics.\n* 🔍 **Discover Equity Insights:** Uncover which traits have the strongest impact on education.")
        st.button("View Data Findings", on_click=go_to_findings)

    with home_col2:
        # Load local hero image based on user specifications
        home_img_path = Path(__file__).parent / "Images" / "Home_Page.jpeg"
        if home_img_path.exists():
            st.image(str(home_img_path), use_container_width=True,
                     caption="Empowering Education through Equity and Data")
        else:
            st.info(f"[Image Placeholder: {home_img_path.name} not found]")


# ==========================================
# PAGE 2: DATA FINDINGS (ANALYSIS)
# ==========================================
elif st.session_state.current_page == "Findings":
    st.title("Data Findings")
    st.write(
        "This section presents our exploratory data analysis, focusing heavily on how socioeconomic markers correlate with test scores.")

    try:
        df = load_student_data()
        df_clean = df.copy()

        # Subsection routing dropdown
        subsection = st.selectbox(
            "Select an analysis to view:",
            ["1. Statistical Summary", "2. Univariate Analysis", "3. Bivariate Analysis",
             "4. Correlation & Impact Analysis", "5. Predictive Modeling"]
        )
        st.markdown("---")
        st.subheader(subsection)

        # --- SUBSECTION 1: STATISTICAL SUMMARY ---
        if subsection == "1. Statistical Summary":
            st.write(
                "The Statistical Summary provides a high-level mathematical overview of our dataset. It computes averages, standard deviations, and quartiles for the numeric scores, while also giving us a precise headcount for all the demographic categories.")

            st.write("#### Numeric Variables Overview")
            numeric_cols = [c for c in ['Age', 'Study Hours', 'Attendance Percentage', 'Math Score', 'Science Score',
                                        'English Score', 'Overall Score'] if c in df.columns]
            if numeric_cols: st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            st.write("#### Categorical Variables Overview (Frequency Distributions)")
            cat_cols = [c for c in ['Gender', 'School Type', 'Parent Education', 'Internet Access', 'Travel Time',
                                    'Extra Activities', 'Study Method', 'Final Grade'] if c in df.columns]

            # Distribute categorical tables across 3 columns
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for i, col in enumerate(cat_cols):
                with cols[i % 3]:
                    st.write(f"**{col}**")
                    st.dataframe(df[col].value_counts(), use_container_width=True)

        # --- SUBSECTION 2: UNIVARIATE ANALYSIS ---
        elif subsection == "2. Univariate Analysis":
            st.write(
                "Univariate Analysis involves examining a single variable at a time. The goal here is to understand the underlying distribution, identify central tendencies, and spot any potential outliers within individual demographics and scores before we start looking for complex relationships.")

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Overall Score Distribution")
                if 'Overall Score' in df_clean.columns:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.histplot(df_clean['Overall Score'].dropna(), kde=True, color='#2E7D6B', ax=ax1)
                    ax1.set_xlabel("Overall Score")
                    st.pyplot(fig1)

            with col2:
                st.write("#### Final Grade Distribution")
                if 'Final Grade' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Final Grade'])
                    valid_order = [g for g in ['A', 'B', 'C', 'D', 'E', 'F'] if g in plot_df['Final Grade'].unique()]
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=plot_df, x='Final Grade', order=valid_order, color='#2E7D6B', ax=ax2)
                    st.pyplot(fig2)

            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                st.write("#### Parental Education Distribution")
                if 'Parent Education' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Parent Education'])
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=plot_df, y='Parent Education', color='#2E7D6B', order=EDU_ORDER, ax=ax3)
                    ax3.set_ylabel("")
                    st.pyplot(fig3)

            with col4:
                st.write("#### Primary Study Method")
                if 'Study Method' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Study Method'])
                    fig4, ax4 = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=plot_df, x='Study Method', color='#2E7D6B',
                                  order=sorted(plot_df['Study Method'].unique()), ax=ax4)
                    st.pyplot(fig4)

        # --- SUBSECTION 3: BIVARIATE ANALYSIS ---
        elif subsection == "3. Bivariate Analysis":
            st.write(
                "Bivariate Analysis compares two different variables to uncover relationships, trends, and potential causalities. Here, we investigate how various demographic factors directly influence total academic scores. You can use the toggles below to switch between different demographics and chart types.")

            # Interactive Plot controls
            col_biv1, col_biv2 = st.columns([2, 1])
            with col_biv1:
                biv_option = st.radio(
                    "Choose demographic comparison:",
                    ["Parent Education vs Overall Score", "Internet Access vs Overall Score",
                     "Study Method vs Overall Score", "Gender vs Overall Score", "Extra Activities vs Overall Score"],
                    horizontal=True
                )
            with col_biv2:
                plot_type = st.radio(
                    "Choose Plot Type:",
                    ["Box Plot", "Violin Plot"],
                    horizontal=True
                )

            if df_clean.empty:
                st.warning("Data is missing or empty.")
            elif 'Overall Score' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                x_col = biv_option.split(" vs ")[0]

                if x_col in df_clean.columns:
                    plot_df = df_clean.dropna(subset=[x_col, 'Overall Score'])
                    order = EDU_ORDER if x_col == 'Parent Education' else None

                    st.write(f"#### Impact of {x_col} on Overall Score")

                    if x_col == 'Parent Education':
                        if plot_type == "Violin Plot":
                            sns.violinplot(data=plot_df, x='Overall Score', y=x_col, color='#2E7D6B', order=order,
                                           ax=ax)
                        else:
                            sns.boxplot(data=plot_df, x='Overall Score', y=x_col, color='#2E7D6B', order=order, ax=ax)
                    else:
                        if plot_type == "Violin Plot":
                            sns.violinplot(data=plot_df, x=x_col, y='Overall Score', color='#2E7D6B', order=order,
                                           ax=ax)
                        else:
                            sns.boxplot(data=plot_df, x=x_col, y='Overall Score', color='#2E7D6B', order=order, ax=ax)

                    st.pyplot(fig)

                    if plot_type == "Violin Plot":
                        st.info(
                            "💡 **Interpretation:** The Violin Plot shows the **density** of scores. The wider the violin section, the more students achieved that specific score.")
                    else:
                        box_interpretations = {
                            "Parent Education vs Overall Score": (
                                "💡 **Interpretation:** Different educational attainments of parents show identical median lines and interquartile ranges (IQR), implying that there is no single parents' educational background that has an advantage over the others."
                            ),
                            "Internet Access vs Overall Score": (
                                "💡 **Interpretation:** The group of students who have access to the internet, and those who do not, showed similar median scores and IQR's, suggesting that internet access has no significant effect on student performance."
                            ),
                            "Study Method vs Overall Score": (
                                "💡 **Interpretation:** Consistency among all study methods as observed by the equal box height, median line, and range of whiskers implies that no specific study method has an academic advantage over the others."
                            ),
                            "Gender vs Overall Score": (
                                "💡 **Interpretation:** Gender has no significant effect on the obtained overall scores of the students. Students from all three groups have similar median scores and similar quartile ranges."
                            ),
                            "Extra Activities vs Overall Score": (
                                "💡 **Interpretation:** Students who participate in extra activities and those who do not have similar median scores and similar quartile ranges, suggesting that there is no advantage in having or not having extra activities."
                            ),
                        }
                        st.info(box_interpretations.get(biv_option, "💡 **Interpretation:** Compare the median lines and IQR boxes across groups to identify performance differences."))
                else:
                    st.error(f"Column '{x_col}' not found in the dataset.")

            st.markdown("---")

            # Plotly Interactive Explorer component
            st.write("#### 🚀 Advanced Interactive Explorer")
            st.write(
                "Use these highly readable, interactive visualizations to truly understand the data structure. You can click on sections to drill down, hover over nodes to see exact averages, and easily compare socioeconomic subgroups.")

            tab1, tab2, tab3 = st.tabs(
                ["🌞 Socioeconomic Sunburst", "🕸️ Subject Performance Radar", "🎻 Split Density Profiles"])

            with tab1:
                st.write(
                    "**Click on any slice to expand it!** The **size** of the wedge shows how many students are in that group. The central slice represents **access to internet**. The **color** represents their Average Overall Score (Dark Red = Lowest, Dark Green = Highest).")
                if all(col in df_clean.columns for col in
                       ['Parent Education', 'Internet Access', 'Study Method', 'Overall Score']):
                    df_sunburst = df_clean.dropna(
                        subset=['Internet Access', 'Parent Education', 'Study Method', 'Overall Score'])
                    df_agg = df_sunburst.groupby(['Internet Access', 'Parent Education', 'Study Method'])[
                        'Overall Score'].agg(['mean', 'count']).reset_index()

                    fig_sunburst = px.sunburst(
                        df_agg,
                        path=['Internet Access', 'Parent Education', 'Study Method'],
                        values='count',
                        color='mean',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=df_clean['Overall Score'].mean(),
                        labels={'mean': 'Average Score', 'count': 'Number of Students'},
                        title="Interactive Socioeconomic Breakdown"
                    )
                    fig_sunburst.update_layout(margin=dict(t=40, l=0, r=0, b=0))
                    st.plotly_chart(fig_sunburst, use_container_width=True)

            with tab2:
                st.write(
                    "**Compare 'Academic Signatures'.** How do different Parental Education levels score across core subjects?")
                if all(col in df_clean.columns for col in
                       ['Math Score', 'English Score', 'Science Score', 'Parent Education']):
                    categories = ['Math Score', 'English Score', 'Science Score']
                    df_radar = df_clean.groupby('Parent Education')[categories].mean().reset_index()

                    fig_radar = go.Figure()
                    for _, row in df_radar.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[row['Math Score'], row['English Score'], row['Science Score']],
                            theta=categories,
                            fill='toself',
                            name=row['Parent Education']
                        ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[40, 90])),
                        showlegend=True,
                        title="Average Subject Scores by Parental Education"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    st.info(
                        "🖱️ **Pro Tip:** Click on the specific education levels in the legend (on the right) to hide or isolate them for a clearer view!")

            with tab3:
                st.write(
                    "**The direct Socioeconomic Split.** This split violin plot isolates Parental Education on the bottom axis, and groups the density distributions cleanly side-by-side by Internet Access.")
                if all(col in df_clean.columns for col in ['Parent Education', 'Overall Score', 'Internet Access']):
                    fig_split = px.violin(
                        df_clean.dropna(subset=['Parent Education', 'Overall Score', 'Internet Access']),
                        x='Parent Education', y='Overall Score', color='Internet Access',
                        category_orders={"Parent Education": EDU_ORDER},
                        title="Score Densities Grouped by Internet Access",
                        box=True
                    )
                    fig_split.update_layout(violinmode='group', plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                                            margin=dict(t=40, l=0, r=0, b=0))
                    st.plotly_chart(fig_split, use_container_width=True)

        # --- SUBSECTION 4: CORRELATION ANALYSIS ---
        elif subsection == "4. Correlation & Impact Analysis":
            st.write(
                "Correlation Analysis quantifies the mathematical relationship between multiple variables. By isolating specific traits using One-Hot Encoding, we can objectively measure which socioeconomic traits act as the strongest drivers of academic success, and which traits indicate a disadvantage.")

            if 'Overall Score' in df_clean.columns:
                st.markdown("### 1. Which Traits Impact Scores the Most?")
                cat_cols = [c for c in
                            ['Gender', 'School Type', 'Parent Education', 'Internet Access', 'Extra Activities',
                             'Study Method', 'Travel Time'] if c in df_clean.columns]

                # Convert categories to numerical flags for pearson correlation
                df_dummies = pd.get_dummies(df_clean[cat_cols].dropna())
                df_corr_data = pd.concat([df_dummies, df_clean['Overall Score']], axis=1).dropna()
                trait_correlations = df_corr_data.corr()['Overall Score'].drop('Overall Score').sort_values()

                col_corr1, col_corr2 = st.columns([2, 1])
                with col_corr1:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    colors = ['#C44E52' if x < 0 else '#2E7D6B' for x in trait_correlations.values]
                    clean_labels = [c.replace('_', ': ').title() for c in trait_correlations.index]

                    sns.barplot(x=trait_correlations.values, y=clean_labels, palette=colors, ax=ax1)
                    ax1.set_xlabel("Pearson Correlation with Overall Score")
                    ax1.axvline(0, color='black', linewidth=1)
                    st.pyplot(fig1)

                with col_corr2:
                    st.info(
                        "💡 **Interpretation:**\n\nGreen Bars correlate positively with higher scores. Red Bars correlate with lower scores. \nAll traits have a Pearson Correlation absolute value of less than 0.02, indicating that no variable in the dataset has a significant effect on the test scores of students. \nOnline videos as a means of studying and group study as a method for studying have the most positive and most negative impacts, respectively. However, they are only relatively better or worse, but their effects are still negligible")
                st.markdown("---")

                st.markdown("### 2. The Compounding Effect of Advantage")
                st.write(
                    "In this analysis, an **'Advantage'** refers to a specific socioeconomic or behavioral trait that provides a statistical edge in a student's educational journey. For this model, we isolated three distinct advantages:")
                st.markdown("""
                * 🌐 **Technological Advantage:** Having active **Internet Access** at home.
                * 🎓 **Educational Advantage:** Having parents with a higher education degree (**Bachelor's, Master's, or PhD**).
                * 🏃 **Engagement Advantage:** Participating in **Extra Activities** outside of standard class time.
                """)
                st.write(
                    "By calculating a 'Total Advantages' score (from 0 to 3) for each student, we can visualize the compounding nature of access and engagement. The chart below demonstrates how baseline academic performance shifts as students accumulate more of these advantages.")

                df_adv = df_clean.copy().dropna(
                    subset=['Internet Access', 'Parent Education', 'Extra Activities', 'Overall Score'])

                # Map specific categorical string responses to binary integers
                df_adv['Adv_Internet'] = (df_adv['Internet Access'] == 'Yes').astype(int)
                df_adv['Adv_Parent'] = df_adv['Parent Education'].isin(
                    ["Bachelor's Degree", "Master's Degree", "PhD"]).astype(int)
                df_adv['Adv_Extra'] = (df_adv['Extra Activities'] == 'Yes').astype(int)
                df_adv['Total Advantages'] = df_adv['Adv_Internet'] + df_adv['Adv_Parent'] + df_adv['Adv_Extra']

                col_corr3, col_corr4 = st.columns([2, 1])
                with col_corr3:
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    sns.boxplot(data=df_adv, x='Total Advantages', y='Overall Score', palette="crest", ax=ax2)
                    ax2.set_xlabel("Number of Advantages (0 = None, 3 = All)")
                    ax2.set_ylabel("Overall Score")
                    st.pyplot(fig2)

                with col_corr4:
                    st.info(
                        "💡 **Interpretation:**\n\nStudents who have an advantage, regardless of how many, have a higher overall score than a student with no advantage. \nThe number of advantages a student has has no significant effect on the overall, as evidenced by a near-consistent median score of students with different numbers of advantages.")
                st.markdown("---")

                st.markdown("### 3. Academic Synergies (Subject Correlations)")
                score_cols = ['Math Score', 'English Score', 'Science Score']
                available_scores = [c for c in score_cols if c in df_clean.columns]

                if len(available_scores) > 1:
                    col_corr5, col_corr6 = st.columns([2, 1])
                    with col_corr5:
                        fig3, ax3 = plt.subplots(figsize=(5, 4))
                        sns.heatmap(df_clean[available_scores].corr(), annot=True, cmap='crest', fmt='.2f', vmin=0.5,
                                    vmax=1, ax=ax3)
                        st.pyplot(fig3)
                    with col_corr6:
                        st.info(
                            "💡 **Interpretation:**\n\nStudents who perform well in one subject also perform well in other subjects, suggesting that all three subjects are equally predictive of each other.")
                st.markdown("---")

                st.markdown("### 4. Trait Impact by Individual Subject")
                st.write("Correlating specific advantages directly with individual subject scores.")
                if all(col in df_adv.columns for col in ['Adv_Internet', 'Adv_Parent', 'Adv_Extra'] + available_scores):
                    trait_subject_corr = df_adv[['Adv_Internet', 'Adv_Parent', 'Adv_Extra'] + available_scores].corr()
                    trait_subject_corr = trait_subject_corr.loc[
                        ['Adv_Internet', 'Adv_Parent', 'Adv_Extra'], available_scores]
                    trait_subject_corr.index = ['Internet Access', 'Parent Degree', 'Extra Activities']

                    col_corr7, col_corr8 = st.columns([2, 1])
                    with col_corr7:
                        fig4, ax4 = plt.subplots(figsize=(6, 3))
                        sns.heatmap(trait_subject_corr, annot=True, cmap='crest', fmt='.2f', vmin=0, vmax=0.5, ax=ax4)
                        st.pyplot(fig4)
                    with col_corr8:
                        st.info(
                            "💡 **Interpretation:**\n\nAll correlation values are zero or near zero, confirming that no factor has a significant impact on each subject's score")

        # --- SUBSECTION 5: PREDICTIVE MODELING ---
        elif subsection == "5. Predictive Modeling":
            st.write(
                "Predictive Modeling uses historical data to train a machine learning algorithm. We utilize an **Optimized Random Forest Classifier** to map complex patterns within the data and predict the **Exact Grade** a new student is likely to achieve.")

            # Load the cached model
            clf, acc, X_cols, X_cat_cols, features_cat = train_model(df_clean)

            if clf is None:
                st.error(
                    "Not enough clean data to train the model. Ensure Final Grade, Study Hours, and demographic columns are present.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Model Accuracy:** {acc * 100:.2f}%")
                    st.write("**Target:** Predicting Exact Grade (A, B, C, D, E, F)")
                    st.info(
                        "ℹ️ **Model Context:** This model relies on a combination of innate demographic traits and active student behaviors (like Study Hours and Attendance). By feeding both into the Random Forest algorithm, we can capture the holistic student profile and accurately predict the expected final grade category.")

                with col2:
                    st.write("#### Top Predictive Features")
                    importances = clf.feature_importances_
                    feat_df = pd.DataFrame({'Feature': X_cols, 'Importance': importances}).sort_values(by='Importance',
                                                                                                       ascending=False).head(
                        5)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.barplot(data=feat_df, x='Importance', y='Feature', color='#2E7D6B', ax=ax)
                    ax.set_xlabel("Importance Score");
                    ax.set_ylabel("")
                    st.pyplot(fig)

                st.markdown("---")
                st.write("#### 🔮 Make a Live Prediction")
                st.write(
                    "Adjust the behavioral and demographic sliders below to see how the Machine Learning algorithm predicts the academic outcome.")

                # Interactive UI controls mapped to model features
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    user_gender = st.selectbox("Gender", df_clean['Gender'].dropna().unique())
                    user_school = st.selectbox("School Type", df_clean['School Type'].dropna().unique())
                    user_internet = st.selectbox("Internet Access", df_clean['Internet Access'].dropna().unique())
                with col_p2:
                    user_pedu = st.selectbox("Parent Education", EDU_ORDER)
                    user_method = st.selectbox("Study Method", df_clean['Study Method'].dropna().unique())
                    user_extra = st.selectbox("Extra Activities", df_clean['Extra Activities'].dropna().unique())
                with col_p3:
                    user_travel = st.selectbox("Travel Time", df_clean['Travel Time'].dropna().unique())
                    user_study_hr = st.slider("Study Hours (Weekly)", min_value=0.0,
                                              max_value=float(df_clean['Study Hours'].max()), value=5.0, step=0.5)
                    user_attend = st.slider("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0,
                                            step=1.0)

                # Process user input exactly as the training data was processed
                if st.button("Predict Exact Grade"):
                    user_data_cat = pd.DataFrame({
                        'Gender': [user_gender],
                        'School Type': [user_school],
                        'Parent Education': [user_pedu],
                        'Internet Access': [user_internet],
                        'Travel Time': [user_travel],
                        'Extra Activities': [user_extra],
                        'Study Method': [user_method]
                    })
                    user_encoded = pd.get_dummies(user_data_cat).reindex(columns=X_cat_cols, fill_value=0)

                    user_data_num = pd.DataFrame({
                        'Study Hours': [user_study_hr],
                        'Attendance Percentage': [user_attend]
                    })

                    # Combine Data and predict
                    user_final = pd.concat([user_encoded, user_data_num], axis=1)
                    user_final = user_final[X_cols]

                    prediction = clf.predict(user_final)[0]
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Custom UI responses based on prediction outputs
                    if prediction == 'A':
                        st.success("🎉 **Prediction:** This student is likely to achieve an **A Grade**!")
                    elif prediction == 'B':
                        st.success("📈 **Prediction:** This student is likely to achieve a **B Grade**.")
                    elif prediction == 'C':
                        st.info("📊 **Prediction:** This student is likely to achieve a **C Grade**.")
                    elif prediction == 'D':
                        st.warning("📉 **Prediction:** This student is likely to achieve a **D Grade**.")
                    elif prediction == 'E':
                        st.warning("⚠️ **Prediction:** This student is likely to achieve an **E Grade**.")
                    else:
                        st.error("❌ **Prediction:** This student is likely to **Fail (F)**.")

    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_Performance.csv' is saved in the 'Dataset' folder.")

# ==========================================
# PAGE 3: RESOURCES & RAW DATA
# ==========================================
elif st.session_state.current_page == "Resources":
    st.title("Resources")
    st.write(
        "Here you can explore the dataset used to generate all insights in this dashboard. Use the filters to slice the data and review the Data Dictionary below to understand what each column represents.")
    st.markdown(
        "🔗 **Data Source:** [Student Performance Dataset on Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/student-performance-dataset?resource=download)\n<br>",
        unsafe_allow_html=True)

    try:
        df = load_student_data()
        filtered_df = df.copy()

        search_query = st.text_input("Search by Student ID:", value="", placeholder="e.g., 100")
        with st.expander("📊 Filter Data by Categories", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_grades = st.multiselect("Final Grade", options=sorted(
                    df['Final Grade'].dropna().unique()) if 'Final Grade' in df.columns else [])
                selected_gender = st.multiselect("Gender", options=sorted(
                    df['Gender'].dropna().unique()) if 'Gender' in df.columns else [])
            with col2:
                selected_school = st.multiselect("School Type", options=sorted(
                    df['School Type'].dropna().unique()) if 'School Type' in df.columns else [])
                selected_internet = st.multiselect("Internet Access", options=sorted(
                    df['Internet Access'].dropna().unique()) if 'Internet Access' in df.columns else [])
            with col3:
                selected_pedu = st.multiselect("Parent Education",
                                               options=EDU_ORDER if 'Parent Education' in df.columns else [])
                selected_method = st.multiselect("Study Method", options=sorted(
                    df['Study Method'].dropna().unique()) if 'Study Method' in df.columns else [])

        # Apply multi-category filtering logic dynamically
        if search_query and 'Student ID' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Student ID'].astype(str).str.contains(search_query, case=False, na=False)]
        if selected_grades and 'Final Grade' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Final Grade'].isin(selected_grades)]
        if selected_gender and 'Gender' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Gender'].isin(selected_gender)]
        if selected_school and 'School Type' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['School Type'].isin(selected_school)]
        if selected_internet and 'Internet Access' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Internet Access'].isin(selected_internet)]
        if selected_pedu and 'Parent Education' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Parent Education'].isin(selected_pedu)]
        if selected_method and 'Study Method' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Study Method'].isin(selected_method)]

        st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Filtered Dataset", data=csv_data,
                           file_name="student_performance_filtered.csv", mime="text/csv")

        st.markdown("---")
        with st.expander("📖 **Dataset Data Dictionary**", expanded=False):
            st.markdown("""
            Below is a breakdown of the columns available in the dataset and what they represent:

            * **Student ID:** Unique identifier for each student.
            * **Age:** Age of the student.
            * **Gender:** Student gender (Male, Female, Other).
            * **School Type:** Indicates if the student attends a Public or Private school.
            * **Parent Education:** Highest education level attained by the student's parents.
            * **Study Hours:** Number of hours the student spends studying per week.
            * **Attendance Percentage:** Percentage of total classes attended.
            * **Internet Access:** Indicates whether the student has internet access at home.
            * **Travel Time:** Estimated time taken to commute to school.
            * **Extra Activities:** Participation in extracurricular activities.
            * **Study Method:** The primary method the student utilizes to study (e.g., Notes, Textbook, Coaching).
            * **Math Score:** Score achieved in Mathematics.
            * **Science Score:** Score achieved in Science.
            * **English Score:** Score achieved in English.
            * **Overall Score:** Aggregate performance score representing total academic standing.
            * **Final Grade:** The academic tier achieved by the student (A, B, C, D, E, F).
            """)
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_Performance.csv' is saved in the 'Dataset' folder.")

# ==========================================
# PAGE 4: ABOUT US
# ==========================================
elif st.session_state.current_page == "About Us":
    st.title("About Us")
    st.write(
        "We are a team of data analysts and researchers passionate about uncovering the hidden narratives within educational data. Our mission is to leverage statistical analysis and machine learning to highlight the critical disparities that affect student success, enabling more equitable educational outcomes.")

    cols = st.columns(5)
    members = [
        {"name": "Gerard Emmanuel Bernabe", "email": "gmbernabe@up.edu.ph", "image": "Gerard.jpg"},
        {"name": "Johnicky Benedict Salvador", "email": "jesalvador@up.edu.ph", "image": "Johnicky.jpg"},
        {"name": "Lance Emerson Arreza", "email": "lnarreza@up.edu.ph", "image": "Lance.jpg"},
        {"name": "Leo Shane Rubino", "email": "lfrubino@up.edu.ph", "image": "Leo.jpg"},
        {"name": "Lynus Aio Miguel de Torres", "email": "lndetorres@up.edu.ph", "image": "Lynus.jpg"},
    ]
    image_base_path = Path("About Us")
    for i, col in enumerate(cols):
        with col:
            img_filename = members[i]["image"]
            try:
                st.image(str(image_base_path / img_filename), use_container_width=True)
            except Exception:
                st.error(f"Missing {img_filename}")
            st.markdown(f"""
                <div style="text-align: center; margin-top: 8px;">
                    <p style="margin-bottom: 2px; font-size: 0.9rem; font-weight: 600; color: var(--text-primary);">{members[i]['name']}</p>
                    <p style="margin-top: 0px; font-size: 0.8rem; color: var(--text-muted);">{members[i]['email']}</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
