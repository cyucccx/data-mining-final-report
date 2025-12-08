import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

print("===== Loading Data =====")

# ============================================================
# 1. Load dataset
# ============================================================
train = pd.read_csv("mental-health-data/train.csv")
test = pd.read_csv("mental-health-data/test.csv")

# ============================================================
# 2. Split X / y
# ============================================================
X = train.drop(columns=["Depression", "id"])
y = train["Depression"]
test_X = test.drop(columns=["id"])
test_id = test["id"]

# ============================================================
# 3. Missing value cleaning
# ============================================================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols_all = X.select_dtypes(include=["object"]).columns

# Fill numeric NA with median
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
test_X[numeric_cols] = test_X[numeric_cols].fillna(X[numeric_cols].median())

# Fill categorical NA with "Unknown"
X[categorical_cols_all] = X[categorical_cols_all].fillna("Unknown")
test_X[categorical_cols_all] = test_X[categorical_cols_all].fillna("Unknown")

print("Missing values handled.")

# ============================================================
# 4. Degree, Sleep, Diet cleaning + Ordinal transform
# ============================================================

def map_degree_raw(val):
    """Map messy Degree strings to ordered buckets."""
    v = str(val).strip().lower()

    if "class 11" in v or "class 12" in v:
        return "HighSchool"
    if "phd" in v or "doctor" in v:
        return "PhD"
    if any(k in v for k in ["m.", "master", "mtech", "m.tech", "mca", "mba",
                            "msc", "m.sc", "mcom", "m.com", "md"]):
        return "Master"
    if any(k in v for k in ["b.", "btech", "b.tech", "bca", "bsc", "b.sc",
                            "bcom", "b.com", "ba", "bba", "be", "barch",
                            "b.arch", "mbbs", "llb"]):
        return "Bachelor"
    return "Other"

def map_sleep_hours(val):
    """Convert messy sleep duration strings into numeric hours."""
    if pd.isna(val):
        return None

    v = str(val).strip().lower()
    v = v.replace("_", " ").replace("hrs", "").replace("hours", "").replace("hour", "")
    v = v.replace("to", "-").replace("–", "-")

    nums = re.findall(r"\d+\.?\d*", v)
    nums = [float(n) for n in nums]

    if "less" in v and nums:
        return max(0, nums[0] - 0.5)
    if "more" in v and nums:
        return nums[0] + 0.5

    if len(nums) == 0:
        return None
    if len(nums) == 1:
        return nums[0]

    return np.mean(nums)

def map_dietary(val):
    """Normalize dietary habits to ordered labels."""
    v = str(val).strip().lower()

    if v in {"nan", "none", ""}:
        return "Unknown"
    if "unhealthy" in v:
        return "Unhealthy"
    if "more" in v and "healthy" in v:
        return "MoreHealthy"
    if "less" in v and "healthy" in v:
        return "Moderate"
    if "moderate" in v:
        return "Moderate"
    if "healthy" in v:
        return "Healthy"

    try:
        num = float(v)
        if num <= 1: return "Unhealthy"
        if num <= 2: return "Moderate"
        return "Healthy"
    except:
        return "Unknown"

def map_gender(val):
    """Map gender to binary: male->1, female->0, other/unknown->-1."""
    v = str(val).strip().lower()
    if "female" in v:
        return 0
    if "male" in v:
        return 1
    return -1

# Apply cleaning
X["Degree_Clean"] = X["Degree"].apply(map_degree_raw)
test_X["Degree_Clean"] = test_X["Degree"].apply(map_degree_raw)

X["Sleep_Hours"] = X["Sleep Duration"].apply(map_sleep_hours)
test_X["Sleep_Hours"] = test_X["Sleep Duration"].apply(map_sleep_hours)

# Fill missing sleep values with median
sleep_median = pd.concat([X["Sleep_Hours"], test_X["Sleep_Hours"]]).median()
X["Sleep_Hours"] = X["Sleep_Hours"].fillna(sleep_median)
test_X["Sleep_Hours"] = test_X["Sleep_Hours"].fillna(sleep_median)

X["Diet_Clean"] = X["Dietary Habits"].apply(map_dietary)
test_X["Diet_Clean"] = test_X["Dietary Habits"].apply(map_dietary)

# Gender binary
X["Gender_Binary"] = X["Gender"].apply(map_gender)
test_X["Gender_Binary"] = test_X["Gender"].apply(map_gender)

print("Sleep, Degree, Dietary cleaned.")

# Define ordinal column order
ord_cols = ["Degree_Clean", "Diet_Clean"]
deg_order = ["HighSchool", "Bachelor", "Master", "PhD", "Other"]
diet_order = ["Unhealthy", "Moderate", "Healthy", "MoreHealthy", "Unknown"]

ordinal_encoder = OrdinalEncoder(
    categories=[deg_order, diet_order],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X[ord_cols] = ordinal_encoder.fit_transform(X[ord_cols])
test_X[ord_cols] = ordinal_encoder.transform(test_X[ord_cols])

# Drop original raw columns
X = X.drop(columns=["Degree", "Sleep Duration", "Dietary Habits", "Gender"])
test_X = test_X.drop(columns=["Degree", "Sleep Duration", "Dietary Habits", "Gender"])

# ============================================================
# 5. Binary Encoding for Yes/No fields (0/1, must NOT scale)
# ============================================================

binary_cols = [
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]

for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0, "Unknown": -1})
    test_X[col] = test_X[col].map({"Yes": 1, "No": 0, "Unknown": -1})

# Working/Student → 0/1 mapping
X["Working_Binary"] = X["Working Professional or Student"].map({
    "Working Professional": 1, "Student": 0, "Unknown": -1
})
test_X["Working_Binary"] = test_X["Working Professional or Student"].map({
    "Working Professional": 1, "Student": 0, "Unknown": -1
})

X = X.drop(columns=["Working Professional or Student"])
test_X = test_X.drop(columns=["Working Professional or Student"])

print("Binary fields encoded (0/1).")

# ============================================================
# 6. One-Hot (none currently, list kept for extensibility)
# ============================================================
categorical_cols = []

# ============================================================
# 7. Numeric fields splitting (to avoid scaling binary columns)
# ============================================================

# All numeric columns
all_numeric = X.select_dtypes(exclude="object").columns.tolist()

# Binary columns MUST NOT be scaled
binary_cols_fixed = ["Have you ever had suicidal thoughts ?",
                     "Family History of Mental Illness",
                     "Working_Binary",
                     "Gender_Binary"]

# Numeric columns that WILL be scaled
numeric_to_scale = [c for c in all_numeric if c not in binary_cols_fixed]

print("Numeric columns to scale:", numeric_to_scale)
print("Binary columns NOT scaled:", binary_cols_fixed)

# ============================================================
# 8. Outlier removal (only scale-supported columns)
# ============================================================
Q1 = X[numeric_to_scale].quantile(0.25)
Q3 = X[numeric_to_scale].quantile(0.75)
IQR = Q3 - Q1

mask = ~((X[numeric_to_scale] < (Q1 - 1.5 * IQR)) |
         (X[numeric_to_scale] > (Q3 + 1.5 * IQR))).any(axis=1)

X = X[mask]
y = y[mask]

print("Outliers removed. Remaining:", X.shape)

# ============================================================
# 9. Train/Valid split
# ============================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 10. Pipeline (Scaling only numeric_to_scale)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_to_scale),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("binary", "passthrough", binary_cols_fixed)
    ],
    remainder="passthrough",
    sparse_threshold=0
)

preprocessor.set_output(transform="pandas")

# ============================================================
# 11. Fit-transform
# ============================================================
X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(test_X)

# ============================================================
# 12. Export
# ============================================================
X_train_processed.to_csv("X_train_processed.csv", index=False)
X_valid_processed.to_csv("X_valid_processed.csv", index=False)
X_test_processed.to_csv("X_test_processed.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_valid.to_csv("y_valid.csv", index=False)
test_id.to_csv("test_id.csv", index=False)

print("\n===== PREPROCESSING DONE (Binary Not Scaled) =====")
