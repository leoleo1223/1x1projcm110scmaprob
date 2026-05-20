import pandas as pd

# Load raw dataset
raw_df = pd.read_csv("Student_Performance (Raw Data).csv")

# Create a copy for cleaning
clean_df = raw_df.copy()

# -----------------------------
# 1. Rename Columns
# -----------------------------
clean_df.columns = [
    "Student ID",
    "Age",
    "Gender",
    "School Type",
    "Parent Education",
    "Study Hours",
    "Attendance Percentage",
    "Internet Access",
    "Travel Time",
    "Extra Activities",
    "Study Method",
    "Math Score",
    "Science Score",
    "English Score",
    "Overall Score",
    "Final Grade"
]

# -----------------------------
# 2. Standardize Text Formatting
# -----------------------------

# Convert text columns to title case
text_columns = [
    "Gender",
    "School Type",
    "Parent Education",
    "Internet Access",
    "Travel Time",
    "Extra Activities",
    "Study Method"
]

for col in text_columns:
    clean_df[col] = clean_df[col].str.title()

# Convert Final Grade to uppercase
clean_df["Final Grade"] = clean_df["Final Grade"].str.upper()

# -----------------------------
# 3. Clean Specific Values
# -----------------------------

# Standardize parent education labels
clean_df["Parent Education"] = clean_df["Parent Education"].replace({
    "Post Graduate": "Master's Degree",
    "Graduate": "Bachelor's Degree",
    "High School": "High School"
})

# Standardize travel time formatting
clean_df["Travel Time"] = clean_df["Travel Time"].replace({
    "<15 Min": "<15 Min",
    "15-30 Min": "15-30 Min",
    "30-60 Min": "30-60 Min",
    ">60 Min": ">60 Min"
})

# -----------------------------
# 4. Remove Duplicate Rows
# -----------------------------
clean_df = clean_df.drop_duplicates()

# -----------------------------
# 5. Handle Missing Values
# -----------------------------
clean_df = clean_df.dropna()

# -----------------------------
# 6. Save Cleaned Dataset
# -----------------------------
clean_df.to_csv("Student_Performance1.csv", index=False)

print("Data cleaning complete.")
print(clean_df.head())