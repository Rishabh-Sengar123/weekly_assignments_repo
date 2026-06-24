import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("ecommerce_500k.csv")

print("Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

print("\nMissing Values (%)")
print((df.isnull().sum() / len(df)) * 100)

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nNumerical Summary")
print(df.describe())

print("\nCategorical Summary")
print(df.describe(include="object"))

num = df.select_dtypes(include=np.number).columns

for c in num:
    print(c, "Skewness =", df[c].skew())

plt.figure(figsize=(8, 5))
sns.histplot(df["final_price"], kde=True)
plt.title("Final Price Distribution")
plt.show()

cat = df.select_dtypes(include="object").columns

for c in cat:
    print(c, "Unique Values =", df[c].nunique())

hc = [c for c in cat if df[c].nunique() > 50]
print("High Cardinality Columns:", hc)

rr = df["is_returned"].mean() * 100
print("Overall Return Rate:", round(rr, 2), "%")

print("\nReturn Reason Distribution")
print(df["return_reason"].value_counts())

print("\nReturns by Category")
print(df.groupby("category")["is_returned"].mean().sort_values(ascending=False))

print("\nReturns by Shipping Method")
print(df.groupby("shipping_method")["is_returned"].mean().sort_values(ascending=False))

ord_cat = [["Bronze", "Silver", "Gold", "Platinum"]]

num_f = [
    "product_base_price", "discount_percentage", "final_price",
    "quantity", "total_amount", "shipping_cost", "days_to_deliver",
    "rating", "num_reviews", "customer_age",
    "customer_lifetime_value", "session_duration_mins",
    "pages_viewed", "clicks_to_purchase"
]

cat_f = [
    "category", "sub_category", "shipping_method",
    "payment_method", "customer_gender",
    "country", "city"
]

ord_f = ["loyalty_tier"]

num_p = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])

cat_p = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("enc", OneHotEncoder(handle_unknown="ignore"))
])

ord_p = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("enc", OrdinalEncoder(categories=ord_cat))
])

pre = ColumnTransformer([
    ("num", num_p, num_f),
    ("cat", cat_p, cat_f),
    ("ord", ord_p, ord_f)
])

print("ColumnTransformer Pipeline Created Successfully")

corr = df[num_f].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

x = df[num_f].fillna(df[num_f].median())
y = df["is_returned"]

mi = mutual_info_classif(x, y, random_state=42)

imp = pd.Series(mi, index=x.columns)
print("\nMutual Information Scores")
print(imp.sort_values(ascending=False))
