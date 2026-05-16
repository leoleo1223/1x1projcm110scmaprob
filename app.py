import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > section, [data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; margin-top: 0 !important; }
    [data-testid="stMainBlockContainer"], .block-container { padding-top: 0 !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; max-width: 100% !important; }

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

    [data-testid="stRadio"] { margin: 0 !important; padding: 0 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] { display: flex; justify-content: flex-start; align-items: center; gap: 28px; padding: 0 !important; min-height: 0 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label svg, [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label { background: transparent !important; padding: 0 !important; margin: 0 !important; min-height: 0 !important; position: relative; }
    [data-testid="stRadio"] div[role="radiogroup"] label p { font-size: 0.95rem; color: var(--nav-text); margin: 0; padding: 4px 0; transition: color 0.2s ease; }
    [data-testid="stRadio"] div[role="radiogroup"] label p::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; background-color: var(--nav-text-active); transform: scaleX(0); transform-origin: center; transition: transform 0.2s ease; }
    [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p, [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: var(--nav-text-active); font-weight: 600; }
    [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p::after, [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p::after { transform: scaleX(1); }
    [data-testid="stRadio"] div[role="radiogroup"] label:hover p { color: var(--nav-text-active); cursor: pointer; }

    .stButton > button, .stDownloadButton > button { background-color: var(--accent-deep) !important; color: #FFFFFF !important; border: 1px solid var(--accent) !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
    .stButton > button *, .stDownloadButton > button * { color: #FFFFFF !important; }
    .stButton > button:hover, .stDownloadButton > button:hover { background-color: var(--accent) !important; border-color: var(--accent-strong) !important; color: #FFFFFF !important; }
    [data-baseweb="select"] > div, .stTextInput input, .stMultiSelect [data-baseweb="select"] > div { background-color: var(--bg-surface) !important; border-color: var(--border-soft) !important; color: var(--text-primary) !important; }
    [data-testid="stDataFrame"] { background-color: var(--bg-surface); border: 1px solid var(--border-soft); border-radius: 8px; }
    .stAlert { background-color: var(--bg-elevated) !important; border: 1px solid var(--border-soft) !important; color: var(--text-primary) !important; }
</style>
"""

st.set_page_config(page_title="Socioeconomic & Academic Analysis", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
sns.set_theme(style="whitegrid", palette="crest")

# Logical progression for Parental Education
EDU_ORDER = ["Some High School", "High School", "Some College", "Associate's Degree", "Bachelor's Degree",
             "Master's Degree"]


@st.cache_data(show_spinner="Fetching and cleaning dataset...")
def fetch_clean_student_data():
    file_path = Path(__file__).parent / "Dataset" / "Student_performance_10k.csv"
    df = pd.read_csv(file_path)

    def clean_ethnicity(val):
        if pd.isna(val): return val
        match = re.search(r'([A-E])', str(val), re.IGNORECASE)
        if match: return f"Group {match.group(1).upper()}"
        return val

    if 'Ethnicity' in df.columns:
        df['Ethnicity'] = df['Ethnicity'].apply(clean_ethnicity)
    if 'Parental Education' in df.columns:
        df['Parental Education'] = df['Parental Education'].str.title().str.replace("'S", "'s", regex=False)

    if 'Math Score' in df.columns: df['Math Score'] = pd.to_numeric(df['Math Score'], errors='coerce')
    if 'Total Score' in df.columns: df['Total Score'] = pd.to_numeric(df['Total Score'], errors='coerce')

    map_dict = {1.0: 'Completed', 0.0: 'None', 1: 'Completed', 0: 'None', '1.0': 'Completed', '0.0': 'None'}
    lunch_dict = {1.0: 'Standard', 0.0: 'Free/Reduced', 1: 'Standard', 0: 'Free/Reduced'}

    if 'Test Preparation' in df.columns: df['Test Preparation'] = df['Test Preparation'].replace(map_dict)
    if 'Lunch' in df.columns: df['Lunch'] = df['Lunch'].replace(lunch_dict)

    return df


if 'current_page' not in st.session_state: st.session_state.current_page = "Home"


def go_to_findings(): st.session_state.current_page = "Findings"


menu_options = ["Home", "Findings", "Resources", "About Us"]
logo_col, menu_col, _spacer = st.columns([1, 10, 1], vertical_alignment="center")

with logo_col:
    st.markdown('<span class="nav-anchor"></span>', unsafe_allow_html=True)
    logo_path = Path(__file__).parent / "Images" / "Logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=60)
    else:
        st.markdown('<div class="nav-logo">LOGO</div>', unsafe_allow_html=True)

with menu_col:
    st.radio("Navigate:", menu_options, key="current_page", horizontal=True, label_visibility="collapsed")

# PAGE: HOME
if st.session_state.current_page == "Home":
    st.title("Socioeconomic Factors vs Academic Performance")
    home_col1, home_col2 = st.columns([1.2, 1], gap="large")

    with home_col1:
        st.write("### Project Introduction")
        st.write(
            "Welcome to the Student Performance Analytics Dashboard. This application explores a 10,000-entry dataset to analyze **the distinct relationship between a student's Socioeconomic Background and their overall Academic Performance**.")
        st.write(
            "By looking at indicators such as Parental Education and Lunch Subsidies, our goal is to highlight educational disparities and build predictive models to better understand how background demographics impact student success.")
        st.write("### What You Can Do")
        st.markdown(
            "* 📊 **Analyze Socioeconomic Trends**\n* 📈 **Visualize Performance Disparities**\n* 🤖 **Predict Academic Outcomes based on Background**\n* 🔍 **Discover Equity Insights**")
        st.button("View Data Findings", on_click=go_to_findings)

    with home_col2:
        home_img_path = Path(__file__).parent / "Images" / "Home_Page.jpeg"
        if home_img_path.exists():
            st.image(str(home_img_path), use_container_width=True,
                     caption="Empowering Education through Equity and Data")
        else:
            st.info("[Image Placeholder: Home_Page.jpeg not found]")

# PAGE: FINDINGS
elif st.session_state.current_page == "Findings":
    st.title("Data Findings")
    st.write(
        "This section presents our exploratory data analysis, focusing heavily on how socioeconomic markers correlate with test scores.")

    try:
        df = fetch_clean_student_data()
        df_clean = df.copy()

        subsection = st.selectbox(
            "Select an analysis to view:",
            ["1. Statistical Summary", "2. Univariate Analysis", "3. Bivariate Analysis",
             "4. Correlation & Impact Analysis", "5. Predictive Modeling"]
        )
        st.markdown("---")
        st.subheader(subsection)

        if subsection == "1. Statistical Summary":
            st.write("#### Numeric Variables Overview")
            numeric_cols = [c for c in ['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'Total Score']
                            if c in df.columns]
            if numeric_cols: st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            st.write("#### Categorical Variables Overview (Frequency Distributions)")
            cat_cols = [c for c in ['Gender', 'Ethnicity', 'Parental Education', 'Lunch', 'Test Preparation', 'Grade']
                        if c in df.columns]
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for i, col in enumerate(cat_cols):
                with cols[i % 3]:
                    st.write(f"**{col}**")
                    st.dataframe(df[col].value_counts(), use_container_width=True)

        elif subsection == "2. Univariate Analysis":
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Total Score Distribution")
                if 'Total Score' in df_clean.columns:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.histplot(df_clean['Total Score'].dropna(), kde=True, color='#2E7D6B', ax=ax1)
                    ax1.set_xlabel("Total Score")
                    st.pyplot(fig1)

            with col2:
                st.write("#### Grade Distribution")
                if 'Grade' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Grade'])
                    valid_order = [g for g in ['A', 'B', 'C', 'D', 'Fail'] if g in plot_df['Grade'].unique()]
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=plot_df, x='Grade', order=valid_order, color='#2E7D6B', ax=ax2)
                    st.pyplot(fig2)

            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                st.write("#### Parental Education Distribution")
                if 'Parental Education' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Parental Education'])
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=plot_df, y='Parental Education', color='#2E7D6B', order=EDU_ORDER, ax=ax3)
                    ax3.set_ylabel("")
                    st.pyplot(fig3)

            with col4:
                st.write("#### Ethnicity Distribution")
                if 'Ethnicity' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Ethnicity'])
                    fig4, ax4 = plt.subplots(figsize=(6, 4))
                    sns.countplot(data=plot_df, x='Ethnicity', color='#2E7D6B',
                                  order=sorted(plot_df['Ethnicity'].unique()), ax=ax4)
                    st.pyplot(fig4)

        elif subsection == "3. Bivariate Analysis":

            # --- Added Option to switch between Box Plot and Violin Plot ---
            col_biv1, col_biv2 = st.columns([2, 1])
            with col_biv1:
                biv_option = st.radio(
                    "Choose demographic comparison:",
                    ["Parental Education vs Total Score", "Lunch vs Total Score", "Ethnicity vs Total Score",
                     "Gender vs Total Score", "Test Preparation vs Total Score"],
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
            elif 'Total Score' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                x_col = biv_option.split(" vs ")[0]

                if x_col in df_clean.columns:
                    plot_df = df_clean.dropna(subset=[x_col, 'Total Score'])

                    # Sort Ordering Logic
                    if x_col == 'Ethnicity':
                        order = sorted(plot_df[x_col].unique())
                    elif x_col == 'Parental Education':
                        order = EDU_ORDER
                    else:
                        order = None

                    st.write(f"#### Impact of {x_col} on Total Score")

                    # --- Toggle Logic for Plot Type ---
                    if x_col == 'Parental Education':
                        # Keep Parental Education horizontal for long label readability
                        if plot_type == "Violin Plot":
                            sns.violinplot(data=plot_df, x='Total Score', y=x_col, color='#2E7D6B', order=order, ax=ax)
                        else:
                            sns.boxplot(data=plot_df, x='Total Score', y=x_col, color='#2E7D6B', order=order, ax=ax)
                    else:
                        if plot_type == "Violin Plot":
                            sns.violinplot(data=plot_df, x=x_col, y='Total Score', color='#2E7D6B', order=order, ax=ax)
                        else:
                            sns.boxplot(data=plot_df, x=x_col, y='Total Score', color='#2E7D6B', order=order, ax=ax)

                    st.pyplot(fig)

                    # Dynamic info text based on plot type
                    if plot_type == "Violin Plot":
                        st.info(
                            "💡 **Interpretation:** The Violin Plot shows the **density** of scores. The wider the violin section, the more students achieved that specific score.")
                    else:
                        st.info(
                            "💡 **Interpretation:** The Box Plot shows the **median** (center line inside the box) and spread (quartiles) of scores to easily compare subgroup performance.")
                else:
                    st.error(f"Column '{x_col}' not found in the dataset.")

        elif subsection == "4. Correlation & Impact Analysis":
            if 'Total Score' in df_clean.columns:

                st.markdown("### 1. Which Traits Impact Scores the Most?")
                cat_cols = [c for c in ['Gender', 'Ethnicity', 'Parental Education', 'Lunch', 'Test Preparation'] if
                            c in df_clean.columns]

                df_dummies = pd.get_dummies(df_clean[cat_cols].dropna())
                df_corr_data = pd.concat([df_dummies, df_clean['Total Score']], axis=1).dropna()
                trait_correlations = df_corr_data.corr()['Total Score'].drop('Total Score').sort_values()

                fig1, ax1 = plt.subplots(figsize=(10, 8))
                colors = ['#C44E52' if x < 0 else '#2E7D6B' for x in trait_correlations.values]
                clean_labels = [c.replace('_', ': ').title() for c in trait_correlations.index]

                sns.barplot(x=trait_correlations.values, y=clean_labels, palette=colors, ax=ax1)
                ax1.set_xlabel("Pearson Correlation with Total Score")
                ax1.axvline(0, color='black', linewidth=1)
                st.pyplot(fig1)
                st.info("💡 Green Bars correlate positively with higher scores. Red Bars correlate with lower scores.")
                st.markdown("---")

                st.markdown("### 2. The Compounding Effect of Socioeconomic Advantage")
                df_adv = df_clean.copy().dropna(
                    subset=['Lunch', 'Parental Education', 'Test Preparation', 'Total Score'])

                df_adv['Adv_Lunch'] = (df_adv['Lunch'] == 'Standard').astype(int)
                df_adv['Adv_Parent'] = df_adv['Parental Education'].isin(
                    ["Bachelor's Degree", "Master's Degree"]).astype(int)
                df_adv['Adv_Prep'] = (df_adv['Test Preparation'] == 'Completed').astype(int)
                df_adv['Total Advantages'] = df_adv['Adv_Lunch'] + df_adv['Adv_Parent'] + df_adv['Adv_Prep']

                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df_adv, x='Total Advantages', y='Total Score', palette="crest", ax=ax2)
                ax2.set_xlabel("Number of Advantages (0 = None, 3 = All)")
                ax2.set_ylabel("Total Academic Score")
                st.pyplot(fig2)
                st.info(
                    "💡 The median score climbs steadily with each additional socioeconomic or support advantage a student has.")
                st.markdown("---")

                st.markdown("### 3. Academic Synergies (Subject Correlations)")
                score_cols = ['Math Score', 'Reading Score', 'Writing Score', 'Science Score']
                available_scores = [c for c in score_cols if c in df_clean.columns]

                if len(available_scores) > 1:
                    fig3, ax3 = plt.subplots(figsize=(7, 5))
                    sns.heatmap(df_clean[available_scores].corr(), annot=True, cmap='crest', fmt='.2f', vmin=0.5,
                                vmax=1, ax=ax3)
                    st.pyplot(fig3)
                    st.info(
                        "💡 Reading and Writing usually have the highest correlation, indicating literacy skills develop closely in tandem.")
                st.markdown("---")

                st.markdown("### 4. Trait Impact by Individual Subject")
                st.write("Correlating specific advantages directly with individual subject scores.")
                if all(col in df_adv.columns for col in ['Adv_Lunch', 'Adv_Parent', 'Adv_Prep'] + available_scores):
                    trait_subject_corr = df_adv[['Adv_Lunch', 'Adv_Parent', 'Adv_Prep'] + available_scores].corr()
                    trait_subject_corr = trait_subject_corr.loc[
                        ['Adv_Lunch', 'Adv_Parent', 'Adv_Prep'], available_scores]
                    trait_subject_corr.index = ['Standard Lunch', 'Parent Degree', 'Test Prep Completed']

                    fig4, ax4 = plt.subplots(figsize=(8, 4))
                    sns.heatmap(trait_subject_corr, annot=True, cmap='crest', fmt='.2f', vmin=0, vmax=0.5, ax=ax4)
                    st.pyplot(fig4)
                    st.info(
                        "💡 Compare the columns to see which subject is most heavily influenced by each demographic trait.")

        elif subsection == "5. Predictive Modeling":
            st.write(
                "Predicting if a student achieves an **A Grade (>=320 Total Score)** based on Demographics and Preparation.")

            if 'Grade' not in df_clean.columns:
                st.error("The 'Grade' column is missing. Cannot build a predictive model.")
            else:
                features = [f for f in ['Gender', 'Test Preparation', 'Ethnicity', 'Parental Education', 'Lunch'] if
                            f in df_clean.columns]
                ml_df = df_clean.dropna(subset=['Grade'] + features).copy()
                ml_df['Is_A_Grade'] = ml_df['Grade'].apply(lambda x: 1 if x == 'A' else 0)

                if ml_df.empty:
                    st.error("Not enough clean data to train the model.")
                else:
                    X = ml_df[features]
                    y = ml_df['Is_A_Grade']
                    X_encoded = pd.get_dummies(X, drop_first=True)
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

                    with st.spinner('Training Random Forest Classifier...'):
                        clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Model Accuracy:** {acc * 100:.2f}%")
                        st.write("**Target:** Predicting 'Grade A'")
                        st.write(f"**Features Used:** {', '.join(features)}")

                    with col2:
                        st.write("#### Top Predictive Features")
                        importances = clf.feature_importances_
                        feat_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances}).sort_values(
                            by='Importance', ascending=False).head(5)
                        fig, ax = plt.subplots(figsize=(5, 3))
                        sns.barplot(data=feat_df, x='Importance', y='Feature', color='#2E7D6B', ax=ax)
                        ax.set_xlabel("Importance Score");
                        ax.set_ylabel("")
                        st.pyplot(fig)

                    st.markdown("---")
                    st.write("#### 🔮 Make a Live Prediction")
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        user_gender = st.selectbox("Select Gender", df_clean['Gender'].dropna().unique())
                        user_prep = st.selectbox("Select Test Preparation",
                                                 df_clean['Test Preparation'].dropna().unique())
                        user_lunch = st.selectbox("Select Lunch", df_clean['Lunch'].dropna().unique())
                    with col_p2:
                        user_ethnicity = st.selectbox("Select Ethnicity", df_clean['Ethnicity'].dropna().unique())
                        user_pedu = st.selectbox("Select Parental Education",
                                                 df_clean['Parental Education'].dropna().unique())

                    if st.button("Predict Grade"):
                        user_data = pd.DataFrame(
                            {'Gender': [user_gender], 'Test Preparation': [user_prep], 'Ethnicity': [user_ethnicity],
                             'Parental Education': [user_pedu], 'Lunch': [user_lunch]})
                        user_encoded = pd.get_dummies(user_data).reindex(columns=X_encoded.columns, fill_value=0)
                        prediction = clf.predict(user_encoded)[0]
                        st.markdown("<br>", unsafe_allow_html=True)
                        if prediction == 1:
                            st.success("🎉 **Prediction:** This student is likely to achieve an **A Grade**!")
                        else:
                            st.warning("📊 **Prediction:** This student is likely to achieve a **Grade B or lower**.")

    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_performance_10k.csv' is saved in the 'Dataset' folder.")

# PAGE: RESOURCES
elif st.session_state.current_page == "Resources":
    st.title("Resources")
    st.markdown(
        "🔗 **Data Source:** [Students Performance 10000 Clean Data EDA on Kaggle](https://www.kaggle.com/datasets/nadeemajeedch/students-performance-10000-clean-data-eda)\n<br>",
        unsafe_allow_html=True)

    try:
        df = fetch_clean_student_data()
        filtered_df = df.copy()

        search_query = st.text_input("Search by Student Number:", value="", placeholder="e.g., std-100")
        with st.expander("📊 Filter Data by Categories", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_grades = st.multiselect("Grade", options=sorted(
                    df['Grade'].dropna().unique()) if 'Grade' in df.columns else [])
                selected_gender = st.multiselect("Gender", options=sorted(
                    df['Gender'].dropna().unique()) if 'Gender' in df.columns else [])
            with col2:
                selected_ethnicity = st.multiselect("Ethnicity", options=sorted(
                    df['Ethnicity'].dropna().unique()) if 'Ethnicity' in df.columns else [])
                selected_lunch = st.multiselect("Lunch", options=sorted(
                    df['Lunch'].dropna().unique()) if 'Lunch' in df.columns else [])
            with col3:
                selected_pedu = st.multiselect("Parental Education",
                                               options=EDU_ORDER if 'Parental Education' in df.columns else [])
                selected_prep = st.multiselect("Test Preparation", options=sorted(
                    df['Test Preparation'].dropna().unique()) if 'Test Preparation' in df.columns else [])

        if search_query and 'Student Number' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Student Number'].astype(str).str.contains(search_query, case=False, na=False)]
        if selected_grades and 'Grade' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Grade'].isin(selected_grades)]
        if selected_gender and 'Gender' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Gender'].isin(selected_gender)]
        if selected_ethnicity and 'Ethnicity' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Ethnicity'].isin(selected_ethnicity)]
        if selected_lunch and 'Lunch' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Lunch'].isin(selected_lunch)]
        if selected_pedu and 'Parental Education' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Parental Education'].isin(selected_pedu)]
        if selected_prep and 'Test Preparation' in filtered_df.columns: filtered_df = filtered_df[
            filtered_df['Test Preparation'].isin(selected_prep)]

        st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Filtered Dataset", data=csv_data,
                           file_name="student_performance_filtered.csv", mime="text/csv")

        st.markdown("---")
        with st.expander("📖 **Dataset Data Dictionary**", expanded=False):
            st.markdown(
                "* **Student Number:** Unique roll number.\n* **Gender:** Student gender.\n* **Ethnicity:** Ethnic groups.\n* **Parental Education:** Educational background of the student's family.\n* **Lunch:** Free/reduced vs standard lunch.\n* **Test Preparation:** Completed vs None.\n* **Math/Science/Reading/Writing Score:** Subject metrics.\n* **Total Score:** Total out of 400.\n* **Grade:** Academic tier achieved.")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_performance_10k.csv' is saved in the 'Dataset' folder.")

# PAGE: ABOUT US
elif st.session_state.current_page == "About Us":
    st.title("About Us")
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
