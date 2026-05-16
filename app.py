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

    /* --- Hide Default Streamlit Elements --- */
    [data-testid="stHeader"], 
    [data-testid="stToolbar"], 
    [data-testid="stDecoration"], 
    [data-testid="stStatusWidget"] { display: none !important; }

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

    /* --- STRICT CUSTOM TOP NAVIGATION MENU --- */
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) { 
        position: sticky; top: 0; z-index: 999; 
        background-color: var(--nav-bg); 
        padding: 0px 20px !important; 
        border-bottom: 1px solid var(--nav-border); 
        border-radius: 8px; 
        gap: 16px !important; 
        align-items: center !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) > div[data-testid="column"] > div,
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) .element-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stColumn"] { width: auto !important; flex: 0 0 auto !important; }
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stColumn"]:last-child { flex: 1 1 auto !important; }

    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stImage"],
    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) [data-testid="stImage"] > div { 
        margin: 0 !important; 
        padding: 0 !important; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        transform: translateY(-4px); 
    }

    div[data-testid="stHorizontalBlock"]:has(.nav-anchor) img { 
        margin: 0 !important; 
        padding: 0 !important; 
    }

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
        background-color: var(--accent-deep) !important; 
        color: #FFFFFF !important; 
        border: 1px solid var(--accent) !important; 
        border-radius: 8px !important; 
        transition: all 0.2s ease !important; 
    }

    .stButton > button *, .stDownloadButton > button * {
        color: #FFFFFF !important;
    }

    .stButton > button:hover, .stDownloadButton > button:hover { 
        background-color: var(--accent) !important; 
        border-color: var(--accent-strong) !important; 
        color: #FFFFFF !important; 
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
    st.markdown('<span class="nav-anchor"></span>', unsafe_allow_html=True)

    logo_path = Path(__file__).parent / "Images" / "Logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=60)
    else:
        st.markdown('<div class="nav-logo">LOGO</div>', unsafe_allow_html=True)

with menu_col:
    st.radio("Navigate:", menu_options, key="current_page", horizontal=True, label_visibility="collapsed")

# ==========================================
# 5. PAGE ROUTING LOGIC
# ==========================================

# --- HOME PAGE ---
if st.session_state.current_page == "Home":
    st.title("Student Performance Analytics")
    home_col1, home_col2 = st.columns([1.2, 1], gap="large")

    with home_col1:
        st.write("### Project Introduction")
        st.write(
            "Welcome to the Student Performance Analytics Dashboard. This application is built to explore, analyze, and predict outcomes based on a 10,000-entry dataset detailing student demographics, preparation, and subject scores."
        )
        st.write("### What You Can Do")
        st.markdown("""
        * 📊 **Analyze Trends**
        * 📈 **Visualize Data**
        * 🤖 **Predict Performance**
        * 🔍 **Discover Insights**
        """)
        st.write("")
        st.button("View Data Findings", on_click=go_to_findings)

    with home_col2:
        home_img_path = Path("images") / "Home_Page.jpeg"
        st.image(str(home_img_path), use_container_width=True, caption="Empowering Education through Data")

# --- FINDINGS PAGE ---
elif st.session_state.current_page == "Findings":
    st.title("Data Findings")
    st.write("This section presents our exploratory data analysis, insights, and predictive models.")

    try:
        df = fetch_clean_student_data()
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
                        ['Gender', 'Ethnicity', 'Parental Education', 'Lunch', 'Test Preparation', 'Grade'] if
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

            # Row 1: Total Score & Grade
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Total Score Distribution")
                if 'Total Score' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Total Score'])
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.histplot(plot_df['Total Score'], kde=True, color='#2E7D6B', ax=ax1)
                    ax1.set_xlabel("Total Score")
                    st.pyplot(fig1)
                    st.info(
                        "💡 **Interpretation:** [Describe the shape of the distribution, e.g., normal, skewed, where the majority of scores lie, and if there are notable outliers.]")

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
                    st.info(
                        "💡 **Interpretation:** [Discuss which grades are most common, the overall pass/fail ratio, and what this suggests about general student performance.]")

            st.markdown("---")

            # Row 2: Parental Education & Ethnicity
            col3, col4 = st.columns(2)
            with col3:
                st.write("#### Parental Education Distribution")
                if 'Parental Education' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Parental Education'])
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    # Plotted horizontally so the long text labels don't overlap
                    sns.countplot(data=plot_df, y='Parental Education', color='#2E7D6B',
                                  order=plot_df['Parental Education'].value_counts().index, ax=ax3)
                    ax3.set_ylabel("")
                    ax3.set_xlabel("Count")
                    st.pyplot(fig3)
                    st.info(
                        "💡 **Interpretation:** Displays the frequency of the highest education level achieved by the students' parents.")

            with col4:
                st.write("#### Ethnicity Distribution")
                if 'Ethnicity' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Ethnicity'])
                    fig4, ax4 = plt.subplots(figsize=(6, 4))
                    # Sorted alphabetically for neatness
                    sns.countplot(data=plot_df, x='Ethnicity', color='#2E7D6B',
                                  order=sorted(plot_df['Ethnicity'].unique()), ax=ax4)
                    ax4.set_xlabel("Ethnicity Groups")
                    st.pyplot(fig4)
                    st.info(
                        "💡 **Interpretation:** Highlights the representation of different ethnic groups within the dataset.")

        # 3. BIVARIATE ANALYSIS
        elif subsection == "3. Bivariate Analysis":
            st.write("Examining relationships between pairs of variables.")

            biv_option = st.radio(
                "Choose comparison:",
                ["Gender vs Total Score", "Test Prep vs Total Score", "Parental Education vs Total Score",
                 "Ethnicity vs Total Score"],
                horizontal=True
            )

            if df_clean.empty:
                st.warning("Data is missing or empty.")
            elif 'Total Score' in df_clean.columns:
                fig, ax = plt.subplots(figsize=(8, 5))

                if biv_option == "Gender vs Total Score" and 'Gender' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Gender', 'Total Score'])
                    sns.boxplot(data=plot_df, x='Gender', y='Total Score', color='#2E7D6B', ax=ax)
                    st.write("#### How does Gender affect Total Score?")

                elif biv_option == "Test Prep vs Total Score" and 'Test Preparation' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Test Preparation', 'Total Score'])
                    # Converted to boxplot as requested
                    sns.boxplot(data=plot_df, x='Test Preparation', y='Total Score', color='#2E7D6B', ax=ax)
                    st.write("#### Impact of completing a Test Preparation Course")

                elif biv_option == "Parental Education vs Total Score" and 'Parental Education' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Parental Education', 'Total Score'])
                    # For longer text labels like parental education, flipping x and y can improve readability
                    sns.boxplot(data=plot_df, x='Total Score', y='Parental Education', color='#2E7D6B', ax=ax)
                    st.write("#### Does Parental Education correlate with higher scores?")

                elif biv_option == "Ethnicity vs Total Score" and 'Ethnicity' in df_clean.columns:
                    plot_df = df_clean.dropna(subset=['Ethnicity', 'Total Score'])
                    sns.boxplot(data=plot_df, x='Ethnicity', y='Total Score', color='#2E7D6B',
                                order=sorted(plot_df['Ethnicity'].unique()), ax=ax)
                    st.write("#### Performance Differences Across Ethnic Groups")

                st.pyplot(fig)
                st.info(
                    f"💡 **Interpretation:** [Provide insights on how the selected variable ({biv_option.split(' vs ')[0]}) impacts Total Score based on the plot medians, quartiles, and spread.]")

        # 4. CORRELATION ANALYSIS
        elif subsection == "4. Correlation Analysis":
            st.write("Determining the strength and direction of relationships between numeric scores.")

            num_cols = [c for c in ['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'Total Score'] if
                        c in df_clean.columns]
            if len(num_cols) > 1:
                numeric_df = df_clean[num_cols].dropna()
                corr_matrix = numeric_df.corr()

                # Original Correlation Views
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("#### Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(corr_matrix, annot=True, cmap='crest', fmt='.2f', ax=ax, vmin=0, vmax=1)
                    st.pyplot(fig)
                    st.info(
                        "💡 **Interpretation:** [Identify which scores are most strongly correlated. For example, do students who score high in reading also score high in writing based on the correlation coefficients?]")

                with col2:
                    if 'Reading Score' in df_clean.columns and 'Writing Score' in df_clean.columns:
                        st.write("#### Reading vs Writing Scatterplot")
                        plot_df = df_clean.dropna(subset=['Reading Score', 'Writing Score'])
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        sns.scatterplot(data=plot_df, x='Reading Score', y='Writing Score', alpha=0.3, color="#2E7D6B",
                                        ax=ax2)
                        st.pyplot(fig2)
                        st.info(
                            "💡 **Interpretation:** [Explain the trend seen in the scatterplot. E.g., a tight upward cluster indicates a strong positive relationship between Reading and Writing skills.]")

                st.markdown("---")

                # Socioeconomic vs Academic Performance
                st.write("#### Socioeconomic vs Academic Performance")
                st.write(
                    "To understand socioeconomic impact, we map the `Lunch` variable (where Standard = 1, Free/Reduced = 0) and correlate it with the students' academic scores.")

                if 'Lunch' in df_clean.columns:
                    socio_df = df_clean.copy()
                    # Creating a numerical proxy for Socioeconomic Status
                    socio_df['Lunch_Num'] = socio_df['Lunch'].map({'Standard': 1, 'Free/Reduced': 0})

                    socio_cols = ['Lunch_Num'] + num_cols
                    # Get correlation strictly with the Lunch_Num feature
                    socio_corr = socio_df[socio_cols].corr()[['Lunch_Num']].drop('Lunch_Num').sort_values(
                        by='Lunch_Num', ascending=False)

                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=socio_corr['Lunch_Num'], y=socio_corr.index, color="#2E7D6B", ax=ax3)
                    ax3.set_xlabel("Correlation Coefficient (with Standard Lunch)")
                    ax3.set_ylabel("Academic Scores")
                    ax3.set_xlim(-0.5, 0.5)  # Setting reasonable limits for correlation

                    # Add vertical line at 0
                    ax3.axvline(0, color='black', linewidth=0.8)
                    st.pyplot(fig3)

                    st.info(
                        "💡 **Interpretation:** A positive correlation implies that having a 'Standard' lunch (often a proxy for relatively higher socioeconomic status) is associated with higher scores. A longer bar signifies a stronger relationship.")
                else:
                    st.warning("The 'Lunch' column is missing; unable to calculate socioeconomic correlation.")

        # 5. PREDICTIVE MODELING
        elif subsection == "5. Predictive Modeling":
            st.write(
                "Predicting whether a student will achieve an **A Grade (>=320 Total Score)** based purely on Demographics and Preparation.")

            if 'Grade' not in df_clean.columns:
                st.error("The 'Grade' column is missing. Cannot build a predictive model.")
            else:
                # 1. Define Features (Including Lunch)
                desired_features = ['Gender', 'Test Preparation', 'Ethnicity', 'Parental Education', 'Lunch']
                features = [f for f in desired_features if f in df_clean.columns]

                ml_df = df_clean.dropna(subset=['Grade'] + features).copy()
                ml_df['Is_A_Grade'] = ml_df['Grade'].apply(lambda x: 1 if x == 'A' else 0)

                if not features:
                    st.error("Missing required columns for predictive modeling.")
                elif ml_df.empty:
                    st.error("Not enough clean data to train the model.")
                else:
                    # 2. Train the Model
                    X = ml_df[features]
                    y = ml_df['Is_A_Grade']

                    X_encoded = pd.get_dummies(X, drop_first=True)
                    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2,
                                                                        random_state=42)

                    with st.spinner('Training Random Forest Classifier...'):
                        clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                    # 3. Display Model Performance
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Model Accuracy:** {acc * 100:.2f}%")
                        st.write("**Target:** Predicting 'Grade A'")
                        st.write("**Model Used:** Random Forest Classifier")
                        st.write(f"**Features Used:** {', '.join(features)}")
                        st.info(
                            "💡 **Model Interpretation:** [Briefly explain if this accuracy means the model is highly reliable or just moderately predictive given the chosen features.]")

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
                        st.info(
                            "💡 **Feature Interpretation:** [Explain which demographic or preparation factors carry the most weight in predicting an 'A' grade according to the model.]")

                    # 4. LIVE PREDICTION TOOL
                    st.markdown("---")
                    st.write("#### 🔮 Make a Live Prediction")
                    st.write("Enter student details below to predict if they will achieve an **A Grade**.")

                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        user_gender = st.selectbox("Select Gender", df_clean['Gender'].dropna().unique())
                        user_prep = st.selectbox("Select Test Preparation",
                                                 df_clean['Test Preparation'].dropna().unique())
                        user_lunch = st.selectbox("Select Lunch", df_clean['Lunch'].dropna().unique())
                    with col_p2:
                        user_ethnicity = st.selectbox("Select Ethnicity",
                                                      df_clean['Ethnicity'].dropna().unique())
                        user_pedu = st.selectbox("Select Parental Education",
                                                 df_clean['Parental Education'].dropna().unique())

                    if st.button("Predict Grade"):
                        # Package inputs
                        user_data = pd.DataFrame({
                            'Gender': [user_gender],
                            'Test Preparation': [user_prep],
                            'Ethnicity': [user_ethnicity],
                            'Parental Education': [user_pedu],
                            'Lunch': [user_lunch]
                        })

                        # Encode inputs to match training data structure
                        user_encoded = pd.get_dummies(user_data)
                        user_encoded = user_encoded.reindex(columns=X_encoded.columns, fill_value=0)

                        # Predict
                        prediction = clf.predict(user_encoded)[0]

                        st.markdown("<br>", unsafe_allow_html=True)
                        if prediction == 1:
                            st.success(
                                "🎉 **Prediction:** Based on the model, this student is likely to achieve an **A Grade**!")
                        else:
                            st.warning(
                                "📊 **Prediction:** Based on the model, this student is likely to achieve a **Grade B or lower**.")

    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Student_performance_10k.csv' is saved in the 'Dataset' folder.")

# --- RESOURCES PAGE ---
elif st.session_state.current_page == "Resources":
    st.title("Resources")
    st.write("Preview of the dataset used for this project: **Student Performance**.")

    try:
        df = fetch_clean_student_data()
        filtered_df = df.copy()

        # 1. Search Bar at the top
        search_query = st.text_input("Search by Student Number:", value="", placeholder="e.g., std-100")

        # 2. Expander for all categorical filters
        with st.expander("📊 Filter Data by Categories (Leave empty to show all)", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'Grade' in df.columns:
                    unique_grades = sorted([g for g in df['Grade'].unique() if pd.notna(g)])
                    selected_grades = st.multiselect("Grade", options=unique_grades, default=[])
                else:
                    selected_grades = []

                if 'Gender' in df.columns:
                    unique_gender = sorted([g for g in df['Gender'].unique() if pd.notna(g)])
                    selected_gender = st.multiselect("Gender", options=unique_gender, default=[])
                else:
                    selected_gender = []

            with col2:
                if 'Ethnicity' in df.columns:
                    unique_ethnicity = sorted([g for g in df['Ethnicity'].unique() if pd.notna(g)])
                    selected_ethnicity = st.multiselect("Ethnicity", options=unique_ethnicity, default=[])
                else:
                    selected_ethnicity = []

                if 'Lunch' in df.columns:
                    unique_lunch = sorted([g for g in df['Lunch'].unique() if pd.notna(g)])
                    selected_lunch = st.multiselect("Lunch", options=unique_lunch, default=[])
                else:
                    selected_lunch = []

            with col3:
                if 'Parental Education' in df.columns:
                    unique_pedu = sorted([g for g in df['Parental Education'].unique() if pd.notna(g)])
                    selected_pedu = st.multiselect("Parental Education", options=unique_pedu, default=[])
                else:
                    selected_pedu = []

                if 'Test Preparation' in df.columns:
                    unique_prep = sorted([g for g in df['Test Preparation'].unique() if pd.notna(g)])
                    selected_prep = st.multiselect("Test Preparation", options=unique_prep, default=[])
                else:
                    selected_prep = []

        # 3. Apply Filters to the Dataframe
        if search_query and 'Student Number' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['Student Number'].astype(str).str.contains(search_query, case=False, na=False)]

        if selected_grades and 'Grade' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Grade'].isin(selected_grades)]

        if selected_gender and 'Gender' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]

        if selected_ethnicity and 'Ethnicity' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Ethnicity'].isin(selected_ethnicity)]

        if selected_lunch and 'Lunch' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Lunch'].isin(selected_lunch)]

        if selected_pedu and 'Parental Education' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Parental Education'].isin(selected_pedu)]

        if selected_prep and 'Test Preparation' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Test Preparation'].isin(selected_prep)]

        # 4. Display Results
        st.caption(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Filtered Dataset (CSV)", data=csv_data,
                           file_name="student_performance_filtered.csv", mime="text/csv")

        st.markdown("---")
        with st.expander("📖 **Dataset Data Dictionary**", expanded=False):
            st.markdown("""
            Below is a breakdown of the columns available in the dataset and what they represent:

            * **Student Number:** Represents the unique roll number of the student.
            * **Gender:** Useful for analyzing performance differences between male and female students.
            * **Ethnicity:** Allows analysis of academic performance trends across different ethnic groups.
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
        {"name": "Gerard Emmanuel Bernabe", "email": "gmbernabe@up.edu.ph", "image": "Gerard.jpg"},
        {"name": "Johnicky Benedict Salvador", "email": "jesalvador@up.edu.ph", "image": "Johnicky.jpg"},
        {"name": "Lance Emerson Arreza", "email": "lnarreza@up.edu.ph", "image": "Lance.jpg"},
        {"name": "Leo Shane Rubino", "email": "lfrubino@up.edu.ph", "image": "Leo.jpg"},
        {"name": "Lynus Aio Miguel de Torres", "email": "lndetorres@up.edu.ph", "image": "Lynus.jpg"},
    ]

    # Define the base path for images
    image_base_path = Path("About Us")

    for i, col in enumerate(cols):
        with col:
            img_filename = members[i]["image"]
            full_img_path = str(image_base_path / img_filename)

            try:
                st.image(full_img_path, use_container_width=True)
            except Exception:
                st.error(f"Missing {img_filename}")

            st.markdown(f"""
                <div style="text-align: center; margin-top: 8px;">
                    <p style="margin-bottom: 2px; font-size: 0.9rem; font-weight: 600; color: var(--text-primary);">
                        {members[i]['name']}
                    </p>
                    <p style="margin-top: 0px; font-size: 0.8rem; color: var(--text-muted);">
                        {members[i]['email']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
