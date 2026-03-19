import pandas as pd

df = pd.read_csv("data/processed/events.csv")

print("Unique users:", df["user_id"].nunique())
print("Total rows:", len(df))

print("Duplicates:",
      df.duplicated(subset=["user_id", "item_id", "timestamp"]).sum())