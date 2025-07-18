import pandas as pd

df = pd.read_csv("wustl-ehms-2020.csv")
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")