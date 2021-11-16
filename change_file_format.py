import pandas as pd

months = ["apr", "may", "jun", "jul", "aug", "sep"]
for month in months:
    df = pd.read_csv(f"data/uber-processed-data-{month}14.csv", encoding="utf-8")
    df.to_csv(f"data/uber-processed-data-{month}14.csv", encoding="utf-8", index=False)
