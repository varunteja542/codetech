import pandas as pd
from datetime import datetime


df = pd.read_csv("people-1000.csv")


df.fillna("Unknown", inplace=True)


df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})


df["Date of birth"] = pd.to_datetime(df["Date of birth"], errors="coerce")
df["Age"] = df["Date of birth"].apply(lambda x: datetime.now().year - x.year if pd.notnull(x) else None)


df.to_csv("cleaned_people_data.csv", index=False)

print("ETL Pipeline Complete. Cleaned data saved as 'cleaned_people_data.csv'.")
