# ============================================
# STUDENT PERFORMANCE DATA ANALYSIS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

print("=" * 60)
print("STUDENT PERFORMANCE DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
display(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
display(df.describe())


# ============================================
# DATA CLEANING
# ============================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert numerical columns to numeric
numeric_columns = [
    "math_score",
    "reading_score",
    "writing_score",
    "attendance"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

# Fill missing numerical values with median
for column in numeric_columns:
    if column in df.columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

# Fill missing categorical values
categorical_columns = [
    "gender",
    "race_ethnicity",
    "parental_education",
    "lunch",
    "test_preparation"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].fillna("Unknown")

# Remove extra spaces from categorical values
for column in categorical_columns:
    if column in df.columns:
        df[column] = (
            df[column].astype(str).str.strip()
        )

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records After Cleaning:")
print(df.duplicated().sum())

print("\nDataset Shape After Cleaning:", df.shape)


# ============================================
# CALCULATE TOTAL AND AVERAGE SCORES
# ============================================

df["total_score"] = (
    df["math_score"]
    + df["reading_score"]
    + df["writing_score"]
)

df["average_score"] = df["total_score"] / 3

print("\nStudent Scores:")

display(
    df[
        [
            "gender",
            "math_score",
            "reading_score",
            "writing_score",
            "total_score",
            "average_score"
        ]
    ].head()
)


# ============================================
# STATISTICAL ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

subjects = {
    "Math": df["math_score"].values,
    "Reading": df["reading_score"].values,
    "Writing": df["writing_score"].values
}

for subject, scores in subjects.items():

    print(f"\n{subject} Score:")

    print("Mean:", round(np.mean(scores), 2))

    print("Median:", round(np.median(scores), 2))

    print(
        "Standard Deviation:",
        round(np.std(scores), 2)
    )

    print("Minimum:", np.min(scores))

    print("Maximum:", np.max(scores))


# ============================================
# SUBJECT-WISE AVERAGE SCORES
# ============================================

subject_average = {
    "Math": df["math_score"].mean(),
    "Reading": df["reading_score"].mean(),
    "Writing": df["writing_score"].mean()
}

print("\nSubject-wise Average Scores:")

for subject, score in subject_average.items():
    print(subject, ":", round(score, 2))

# Bar chart
plt.figure(figsize=(8, 5))

plt.bar(
    subject_average.keys(),
    subject_average.values()
)

plt.title("Subject-wise Average Scores")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.ylim(0, 100)

plt.show()


# ============================================
# DISTRIBUTION OF AVERAGE STUDENT SCORES
# ============================================

plt.figure(figsize=(9, 5))

sns.histplot(
    df["average_score"],
    bins=20,
    kde=True
)

plt.title("Distribution of Average Student Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")

plt.show()


# ============================================
# GENDER-WISE PERFORMANCE
# ============================================

gender_performance = df.groupby("gender")[
    [
        "math_score",
        "reading_score",
        "writing_score"
    ]
].mean()

print("\nGender-wise Performance:")

display(gender_performance.round(2))

gender_performance.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Gender-wise Subject Performance")
plt.xlabel("Gender")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.ylim(0, 100)

plt.show()


# ============================================
# CATEGORY-WISE PERFORMANCE
# ============================================

category_performance = df.groupby("race_ethnicity")[
    [
        "math_score",
        "reading_score",
        "writing_score",
        "average_score"
    ]
].mean()

print("\nCategory-wise Performance:")

display(category_performance.round(2))


# ============================================
# ATTENDANCE AND AVERAGE SCORE ANALYSIS
# ============================================

if "attendance" in df.columns:

    print("\nAttendance and Average Score Analysis:")

    print(
        "Attendance Mean:",
        round(df["attendance"].mean(), 2)
    )

    print(
        "Average Score Mean:",
        round(df["average_score"].mean(), 2)
    )

    # Scatter plot
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="attendance",
        y="average_score"
    )

    plt.title("Attendance vs Average Student Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Average Score")

    plt.show()


# ============================================
# CORRELATION ANALYSIS
# ============================================

corr_cols = [
    "math_score",
    "reading_score",
    "writing_score",
    "average_score"
]

if "attendance" in df.columns:
    corr_cols.insert(0, "attendance")

correlation = df[corr_cols].corr()

print("\nCorrelation Matrix:")

display(correlation.round(2))

# Correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================
# DISTRIBUTION OF SUBJECT SCORES
# ============================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df[
        [
            "math_score",
            "reading_score",
            "writing_score"
        ]
    ]
)

plt.title("Distribution of Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Score")

plt.show()


# ============================================
# TOP 10 STUDENTS
# ============================================

top_students = df.sort_values(
    by="average_score",
    ascending=False
).head(10)

print("\nTop 10 Students:")

display(
    top_students[
        [
            "gender",
            "math_score",
            "reading_score",
            "writing_score",
            "average_score"
        ]
    ].round(2)
)


# ============================================
# STUDENTS WITH HIGH ATTENDANCE
# ============================================

if "attendance" in df.columns:

    high_attendance = df[
        df["attendance"] >= 90
    ]

    print("\nStudents with Attendance >= 90:")

    display(high_attendance.head())


# ============================================
# HIGH-PERFORMANCE STUDENTS
# ============================================

high_performance = df[
    df["average_score"] >= 80
]

print("\nStudents with Average Score >= 80:")

display(high_performance.head())


# ============================================
# FINAL PROJECT SUMMARY
# ============================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("Total Students:", len(df))

print(
    "Average Math Score:",
    round(df["math_score"].mean(), 2)
)

print(
    "Average Reading Score:",
    round(df["reading_score"].mean(), 2)
)

print(
    "Average Writing Score:",
    round(df["writing_score"].mean(), 2)
)

print(
    "Overall Average Score:",
    round(df["average_score"].mean(), 2)
)

if "attendance" in df.columns:

    print(
        "Average Attendance:",
        round(df["attendance"].mean(), 2)
    )

print(
    "Highest Average Score:",
    round(df["average_score"].max(), 2)
)

print(
    "Lowest Average Score:",
    round(df["average_score"].min(), 2)
)

print("\nAnalysis Completed Successfully!")