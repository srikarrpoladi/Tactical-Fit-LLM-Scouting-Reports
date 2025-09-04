import pandas as pd
import re

df = pd.read_csv("/Users/srikarpoladi/Downloads/Fifa25data/data/raw/male_players.csv")

def cleaneded(df):
    df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", "url"], errors="ignore")
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace(r"[^\w_]", "", regex=True)

    if "play_style" in df.columns:
        df["play_style"] = df["play_style"].fillna("None")
        
    return df

df_clean = cleaneded(df)
df_clean.to_csv("/Users/srikarpoladi/Downloads/Fifa25data/data/processed/fifa25_clean.csv", index=False)



    