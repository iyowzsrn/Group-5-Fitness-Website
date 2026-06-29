import pandas as pd

# =====================================
# 1. LOAD DATASET
# =====================================

df = pd.read_csv("gym_members_exercise_tracking_synthetic_data.csv")

print("Original Dataset Shape:", df.shape)

# =====================================
# 2. HANDLE MISSING VALUES
# =====================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove rows containing missing values

rows_before = len(df)

df = df.dropna()

rows_after = len(df)

print("\nRows Removed Due to Missing Values:",
      rows_before - rows_after)

# =====================================
# 3. REMOVE DUPLICATES
# =====================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows Found:", duplicates)

df.drop_duplicates(inplace=True)

# =====================================
# 4. CORRECT INCONSISTENT ENTRIES
# =====================================

# Standardize text values

df["Gender"] = df["Gender"].str.strip().str.title()
df["Workout_Type"] = df["Workout_Type"].str.strip().str.title()

df["Gender"] = (
    df["Gender"]
    .str.strip()
    .str.title()
)

df["Workout_Type"] = (
    df["Workout_Type"]
    .astype(str)
    .str.strip()
    .str.replace(r"\\[A-Za-z]*", "", regex=True)
    .str.replace("\\", "", regex=False)
    .replace(r'^\s*$', pd.NA, regex=True)
)

df = df.dropna(subset=["Workout_Type"])

# =====================================
# 5. RENAME COLUMNS
# =====================================

df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_per_")
)

print("\nUpdated Column Names:")
print(df.columns)

# =====================================
# 6. CONVERT DATA TYPES
# =====================================

# Convert Max BPM to numeric if needed

df["max_bpm"] = pd.to_numeric(df["max_bpm"], errors="coerce")

# Fill values that became missing after conversion

df["max_bpm"] = df["max_bpm"].fillna(df["max_bpm"].median())

# =====================================
# 7. FILTER INVALID DATA
# =====================================

# Remove impossible ages

df = df[(df["age"] >= 10) & (df["age"] <= 100)]

# Remove invalid heights

df = df[(df["height_m"] >= 1.0) & (df["height_m"] <= 2.5)]

# Remove invalid weights

df = df[df["weight_kg"] > 0]

# Remove negative calories

df = df[df["calories_burned"] >= 0]

# Remove invalid BMI

df = df[df["bmi"] > 0]

# =====================================
# 8. HANDLE OUTLIERS (IQR METHOD)
# =====================================

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    df = df[
        (df[col] >= lower_limit) &
        (df[col] <= upper_limit)
    ]

# =====================================
# 10. FINAL CHECK
# =====================================

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:", df.shape)

# =====================================
# 11. SAVE CLEANED DATASET
# =====================================

df.to_csv("fitness_tracker_cleaned.csv", index=False)

print("\nCleaning Completed Successfully!")
print("New file created: fitness_tracker_cleaned.csv")