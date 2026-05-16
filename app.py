import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. CSS STYLING (CLEANED UP & ORGANIZED)
# ==========================================
CUSTOM_CSS = """
<style>
    /* --- Color Variables --- */
    :root {
        --bg-app: #EAF5F0;
        --bg-surface: #FFFFFF;
        --bg-elevated: #F1F8F4;
        --border-soft: #C9DDD3;
        --text-primary: #14352C;
        --text-muted: #4F6B62;
        --accent: #2E7D6B;
        --accent-strong: #5FE0BE;
        --accent-deep: #0E5C4A;
        --nav-bg: #0E2A23;
        --nav-text: #B6D8CC;
        --nav-text-active: #5FE0BE;
        --nav-border: #1B4D40;
    }

    /* --- Global App Styles --- */
    .stApp { background-color: var(--bg-app); color: var(--text-primary); }
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label, 
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: var(--text-primary); }
    .stApp a { color: var(--accent-deep); }
    hr { border-color: var(--border-soft) !important; margin: 6px 0 14px 0 !important; }
    
    /* --- Hide Default Streamlit Elements ---
    [data-testid="stHeader"], 
    [data-testid="stToolbar"], 
    [data-testid="stDecoration"], 
    [data-testid="stStatusWidget"] { display: none !important; } */

    /* --- Layout Padding Resets --- */
    [data-testid="stAppViewContainer"], 
    [data-testid="stAppViewContainer"] > section, 
    [data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; margin-top: 0 !important; }

    [data-testid="stMainBlockContainer"], .block-container { 
        padding-top: 0 !important; 
        padding-left: 2.5rem !important; 
        padding-right: 2.5rem !important; 
        max-width: 100% !important; 
    }

    /* --- Custom Top Navigation Menu --- */
    div[data-testid="stHorizontalBlock"]:has(.nav-logo) { 
        position: sticky; top: 0; z-index: 999; 
        background-color: var(--nav-bg); padding: 10px 16px; 
        border-bottom: 1px solid var(--nav-border); border-radius: 8px; gap: 12px !important; 
    }
    div[data-testid="stHorizontalBlock"]:has(.nav-logo) [data-testid="stColumn"] { width: auto !important; flex: 0 0 auto !important; }
    div[data-testid="stHorizontalBlock"]:has(.nav-logo) [data-testid="stColumn"]:last-child { flex: 1 1 auto !important; }

    .nav-logo { 
        display: flex; align-items: center; justify-content: center; 
        width: 36px; height: 36px; border-radius: 50%; 
        background-color: var(--accent-strong); color: var(--nav-bg); 
        font-weight: 800; font-size: 0.75rem; letter-spacing: 1px; flex-shrink: 0; 
    }

    /* --- Streamlit Radio Button Overrides (Menu Tabs) --- */
    [data-testid="stRadio"] { margin: 0 !important; padding: 0 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] { display: flex; justify-content: flex-start; align-items: center; gap: 28px; padding: 0 !important; min-height: 0 !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label svg, 
    [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
    [data-testid="stRadio"] div[role="radiogroup"] label { background: transparent !important; padding: 0 !important; margin: 0 !important; min-height: 0 !important; position: relative; }
    [data-testid="stRadio"] div[role="radiogroup"] label p { font-size: 0.95rem; color: var(--nav-text); margin: 0; padding: 4px 0; transition: color 0.2s ease; }

    /* Menu Underline Animation */
    [data-testid="stRadio"] div[role="radiogroup"] label p::after { 
        content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; 
        background-color: var(--nav-text-active); transform: scaleX(0); 
        transform-origin: center; transition: transform 0.2s ease; 
    }
    [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p, 
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color: var(--nav-text-active); font-weight: 600; }
    [data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"] p::after, 
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p::after { transform: scaleX(1); }
    [data-testid="stRadio"] div[role="radiogroup"] label:hover p { color: var(--nav-text-active); cursor: pointer; }

    /* --- Buttons and Inputs --- */
    .stButton > button, .stDownloadButton > button { 
        background-color: var(--accent-deep); color: #FFFFFF; 
        border: 1px solid var(--accent); border-radius: 8px; transition: all 0.2s ease; 
    }
    .stButton > button:hover, .stDownloadButton > button:hover { 
        background-color: var(--accent); border-color: var(--accent-strong); color: #FFFFFF; 
    }
    [data-baseweb="select"] > div, .stTextInput input, .stMultiSelect [data-baseweb="select"] > div { 
        background-color: var(--bg-surface) !important; border-color: var(--border-soft) !important; color: var(--text-primary) !important; 
    }

    /* --- Tables and Alerts --- */
    [data-testid="stDataFrame"] { background-color: var(--bg-surface); border: 1px solid var(--border-soft); border-radius: 8px; }
    .stAlert { background-color: var(--bg-elevated) !important; border: 1px solid var(--border-soft) !important; color: var(--text-primary) !important; }
</style>
"""

# ==========================================
# 2. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Data Analysis Application", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Set global seaborn styling
sns.set_theme(style="whitegrid", palette="crest")


# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data(show_spinner="Fetching and cleaning dataset...")
def fetch_clean_student_data():
    file_path = Path(__file__).parent / "Dataset" / "Student_performance_10k.csv"
    df = pd.read_csv(file_path)

    # Ensure numeric types are strictly numeric
    if 'Math Score' in df.columns:
        df['Math Score'] = pd.to_numeric(df['Math Score'], errors='coerce')
    if 'Total Score' in df.columns:
        df['Total Score'] = pd.to_numeric(df['Total Score'], errors='coerce')

    # Use .replace() instead of .map() to preserve rows that have unexpected strings/Nans
    map_dict = {1.0: 'Completed', 0.0: 'None', 1: 'Completed', 0: 'None', '1.0': 'Completed', '0.0': 'None'}
    lunch_dict = {1.0: 'Standard', 0.0: 'Free/Reduced', 1: 'Standard', 0: 'Free/Reduced'}

    if 'Test Preparation' in df.columns:
        df['Test Preparation'] = df['Test Preparation'].replace(map_dict)
    if 'Lunch' in df.columns:
        df['Lunch'] = df['Lunch'].replace(lunch_dict)

    return df


# ==========================================
# 4. NAVIGATION STATE & TOP BAR
# ==========================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"


def go_to_findings():
    st.session_state.current_page = "Findings"


menu_options = ["Home", "Findings", "Resources", "About Us"]
logo_col, menu_col, _spacer = st.columns([1, 10, 1], vertical_alignment="center")

with logo_col:
    st.markdown('<div class="nav-logo">LOGO</div>', unsafe_allow_html=True)

with menu_col:
    st.radio("Navigate:", menu_options, key="current_page", horizontal=True, label_visibility="collapsed")

# ==========================================
# 5. PAGE ROUTING LOGIC
# ==========================================

# --- HOME PAGE ---
if st.session_state.current_page == "Home":
    st.title("Student Performance Analytics")
    st.write("### Project Introduction")
    st.write(
        "Welcome to the Student Performance Analytics Dashboard. This application is built to explore, analyze, and predict outcomes based on a 10,000-entry dataset detailing student demographics, preparation, and subject scores.")
    st.button("View Data Findings", on_click=go_to_findings)

# --- FINDINGS PAGE ---
elif st.session_state.current_page == "Findings":
    st.title("Data Findings")
    st.write("This section presents our exploratory data analysis, insights, and predictive models.")

    try:
        df = fetch_clean_student_data()

        # We no longer drop ALL NAs globally to prevent the dataframe from emptying out completely.
        df_clean = df.copy()

        subsection = st.selectbox(
            "Select an analysis to view:",
            [
                "1. Statistical Summary",
                "2. Univariate Analysis",
                "3. Bivariate Analysis",
                "4. Correlation Analysis",
                "5. Predictive Modeling",
            ],
        )
        st.markdown("---")
        st.subheader(subsection)

        # 1. STATISTICAL SUMMARY
        if subsection == "1. Statistical Summary":
            st.write("#### Numeric Variables Overview")
            numeric_cols = [c for c in ['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'Total Score']
                            if c in df.columns]
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            st.write("#### Categorical Variables Overview (Frequency Distributions)")
            cat_cols = [c for c in
                        ['Gender', 'Race/Ethnicity', 'Parental Education', 'Lunch', 'Test Preparation', 'Grade'] if
                        c in df.columns]

            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for i, col in enumerate(cat_cols):
                with cols[i % 3]:
                    st.write(f"**{col}**")
                    st.dataframe(df[col].value_counts(), use_container_width=True)

        # 2. UNIVARIATE ANALYSIS
        elif subsection == "2. Univariate Analysis":
            st.write("Analyzing the distribution of single variables to understand spread and centrality.")

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Total Score Distribution")
                if 'Total Score' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Total Score'])
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.histplot(plot_df['Total Score'], kde=True, color='#2E7D6B', ax=ax1)
                    ax1.set_xlabel("Total Score")
                    st.pyplot(fig1)

            with col2:
                st.write("#### Grade Distribution")
                if 'Grade' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Grade'])
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    grade_order = ['A', 'B', 'C', 'D', 'Fail']
                    valid_order = [g for g in grade_order if g in plot_df['Grade'].unique()]
                    sns.countplot(data=plot_df, x='Grade', order=valid_order, color='#2E7D6B', ax=ax2)
                    ax2.set_xlabel("Grade")
                    st.pyplot(fig2)

        # 3. BIVARIATE ANALYSIS
        elif subsection == "3. Bivariate Analysis":
            st.write("Examining relationships between pairs of variables.")

            biv_option = st.radio(
                "Choose comparison:",
                ["Gender vs Total Score", "Test Prep vs Total Score", "Parental Education vs Total Score"],
                horizontal=True
            )

            if df_clean.empty:
                st.warning("Data is missing or empty.")
            elif 'Total Score' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(8, 5))

                if biv_option == "Gender vs Total Score" and 'Gender' in df_clean.columns:
                    # Drop NAs ONLY for the specific columns we are plotting
                    plot_df = df_clean.dropna(subset=['Gender', 'Total Score'])
                    sns.boxplot(data=plot_df, x='Gender', y='Total Score', color='#2E7D6B', ax=ax)
                    st.write("#### How does Gender affect Total Score?")

                elif biv_option == "Test Prep vs Total Score" and 'Test Preparation' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Test Preparation', 'Total Score'])
                    sns.violinplot(data=plot_df, x='Test Preparation', y='Total Score', color='#2E7D6B', ax=ax)
                    st.write("#### Impact of completing a Test Preparation Course")

                elif biv_option == "Parental Education vs Total Score" and 'Parental Education' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Parental Education', 'Total Score'])
                    sns.boxplot(data=plot_df, x='Total Score', y='Parental Education', color='#2E7D6B', ax=ax)
                    st.write("#### Does Parental Education correlate with higher scores?")

                st.pyplot(fig)

        # 4. CORRELATION ANALYSIS
        elif subsection == "4. Correlation Analysis":
            st.write("Determining the strength and direction of relationships between numeric scores.")

            num_cols = [c for c in ['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'Total Score'] if
                        c in df_clean.columns]
            if len(num_cols) > 1:
                numeric_df = df_clean[num_cols].dropna()
                corr_matrix = numeric_df.corr()

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("#### Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(corr_matrix, annot=True, cmap='crest', fmt='.2f', ax=ax, vmin=0, vmax=1)
                    st.pyplot(fig)

                with col2:
                    if 'Reading Score' in df_clean.columns and 'Writing Score' in df_clean.columns:
                        st.write("#### Reading vs Writing Scatterplot")
                        plot_df = df_clean.dropna(subset=['Reading Score', 'Writing Score'])
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        sns.scatterplot(data=plot_df, x='Reading Score', y='Writing Score', alpha=0.3, color="#2E7D6B",
                                        ax=ax2)
                        st.pyplot(fig2)

        # 5. PREDICTIVE MODELING
        elif subsection == "5. Predictive Modeling":
            st.write(
                "Predicting whether a student will achieve an **A Grade (>=320 Total Score)** based purely on Demographics and Preparation.")

            if 'Grade' not in df_clean.columns:
                st.error("The 'Grade' column is missing. Cannot build a predictive model.")
            else:
                desired_features = ['Gender', 'Test Preparation', 'Race/Ethnicity', 'Parental Education']
                features = [f for f in desired_features if f in df_clean.columns]

                # Clean dataframe specifically for the ML model to avoid blank rows
                ml_df = df_clean.dropna(subset=['Grade'] + features).copy()
                ml_df['Is_A_Grade'] = ml_df['Grade'].apply(lambda x: 1 if x == 'A' else 0)

                if not features:
                    st.error("Missing required columns for predictive modeling.")
                elif ml_df.empty:
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
                        st.write("**Model Used:** Random Forest Classifier")
                        st.write(f"**Features Used:** {', '.join(features)}")

                    with col2:
                        st.write("#### Top Predictive Features")
                        importances = clf.feature_importances_
                        feat_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
                        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(5)

                        fig, ax = plt.subplots(figsize=(5, 3))
                        sns.barplot(data=feat_df, x='Importance', y='Feature', color='#2E7D6B', ax=ax)
                        ax.set_xlabel("Importance Score")
                        ax.set_ylabel("")
                        st.pyplot(fig)

    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_performance_10k.csv' is saved in the 'Dataset' folder.")

# --- RESOURCES PAGE ---
elif st.session_state.current_page == "Resources":
    st.title("Resources")
    st.write("Preview of the dataset used for this project: **Student Performance**.")

    try:
        df = fetch_clean_student_data()
        filtered_df = df.copy()

        filter_col1, filter_col2 = st.columns([2, 3])
        with filter_col1:
            if 'Grade' in df.columns:
                unique_grades = sorted([g for g in df['Grade'].unique() if pd.notna(g)])
                selected_grades = st.multiselect("Filter by Grade (Leave empty to show all):", options=unique_grades,
                                                 default=[])
            else:
                selected_grades = []

        with filter_col2:
            search_query = st.text_input("Search by Student Number:", value="", placeholder="e.g., std-100")

        if len(selected_grades) > 0 and 'Grade' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Grade'].isin(selected_grades)]

        if search_query and 'Student Number' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['Student Number'].astype(str).str.contains(search_query, case=False, na=False)]

        st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Filtered Dataset (CSV)", data=csv_data,
                           file_name="student_performance_filtered.csv", mime="text/csv")

        st.markdown("---")
        with st.expander("📖 **Dataset Data Dictionary**", expanded=True):
            st.markdown("""
            Below is a breakdown of the columns available in the dataset and what they represent:

            * **Student Number:** Represents the unique roll number of the student.
            * **Gender:** Useful for analyzing performance differences between male and female students.
            * **Race/Ethnicity:** Allows analysis of academic performance trends across different racial or ethnic groups.
            * **Parental Education:** Indicates the educational background of the student's family.
            * **Lunch:** Shows whether students receive a free or reduced lunch, which is often a socioeconomic indicator.
            * **Test Preparation:** Tells whether students completed a test preparation course, which could impact their performance.
            * **Math Score:** Provides a measure of each student’s performance in math, used to calculate averages or trends.
            * **Science Score:** Evaluates students' Science knowledge, which can be analyzed to assess overall scientific understanding.
            * **Reading Score:** Measures performance in reading, allowing for insights into literacy and comprehension levels.
            * **Writing Score:** Evaluates students' writing skills, which can be analyzed to assess overall literacy and expression.
            * **Total Score:** Shows the total number achieved by the student out of 400.
            * **Grade:** The academic tier achieved by the student, calculated as follows:
                * **A:** Total marks ≥ 320
                * **B:** Total marks ≥ 250
                * **C:** Total marks ≥ 200
                * **D:** Total marks ≥ 150
                * **Fail:** Total marks < 150
            """)

    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_performance_10k.csv' is saved in the 'Dataset' folder.")

# --- ABOUT US PAGE ---
elif st.session_state.current_page == "About Us":
    st.title("About Us")
    st.write("Meet the team behind this analysis application.")

    cols = st.columns(5)
    members = [
        {"name": "Gerard Emmanuel Bernabe", "email": "gmbernabe@up.edu.ph"},
        {"name": "Johnicky Benedict Salvador", "email": "jesalvador@up.edu.ph"},
        {"name": "Lance Emerson Arreza", "email": "lnarreza@up.edu.ph"},
        {"name": "Leo Shane Rubino", "email": "lfrubino@up.edu.ph"},
        {"name": "Lynus Aio Miguel de Torres", "email": "lndetorres@up.edu.ph"},
    ]
    for i, col in enumerate(cols):
        with col:
            st.image("https://via.placeholder.com/200", caption="Profile Picture", use_container_width=True)
            st.markdown(f"**{members[i]['name']}**")
            st.caption(members[i]["email"])

st.markdown('</div>', unsafe_allow_html=True)
