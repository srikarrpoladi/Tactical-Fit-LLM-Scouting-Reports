import pandas as pd
import re

df = pd.read_csv("/Users/srikarpoladi/Downloads/Fifa25data/data/processed/fifa25_clean.csv")


# --- 1. Combine main + alternative positions ---
def create_all_positions(df):
    if "position" in df.columns and "alternative_positions" in df.columns:
        df["all_positions"] = df.apply(
            lambda row: row["position"] if pd.isna(row["alternative_positions"]) 
            else f"{row['position']}, {row['alternative_positions']}",
            axis=1
        )
    return df

# --- 2. Multi-hot encode positions ---
def encode_positions(df):
    multi_hot_encoded = df["all_positions"].str.get_dummies(sep=", ")
    df = df.drop("all_positions", axis=1).join(multi_hot_encoded)
    
    df["num_positions"] = multi_hot_encoded.sum(axis=1)
    return df

# --- 3. Assign players to position groups ---
def assign_position_group(df):
    def group_position(pos):
        forwards = ["ST", "CF", "LW", "RW"]
        midfielders = ["CM", "CAM", "CDM", "LM", "RM"]
        defenders = ["CB", "LB", "RB", "LWB", "RWB"]
        if pos in forwards:
            return "forward"
        elif pos in midfielders:
            return "midfielder"
        elif pos in defenders:
            return "defender"
        else:
            return "goalkeeper"
    
    df["position_group"] = df["position"].apply(group_position)
    return df

def compute_composite_attributes(df):
    attack_cols = ["finishing", "shot_power", "positioning", "long_shots", "volleys", "penalties"]
    df["attack"] = df[attack_cols].mean(axis=1)
    
    passing_cols = ["vision", "short_passing", "long_passing", "crossing", "curve", "free_kick_accuracy"]
    df["passing"] = df[passing_cols].mean(axis=1)
    
    dribbling_cols = ["dribbling", "agility", "balance", "reactions", "ball_control"]
    df["dribbling"] = df[dribbling_cols].mean(axis=1)
    
    defense_cols = ["interceptions", "standing_tackle", "sliding_tackle", "def_awareness", "heading_accuracy"]
    df["defense"] = df[defense_cols].mean(axis=1)
    
    physical_cols = ["stamina", "strength", "sprint_speed", "jumping", "aggression", "composure"]
    df["physical"] = df[physical_cols].mean(axis=1)
    
    gk_cols = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
    if all(col in df.columns for col in gk_cols):
        df["goalkeeping"] = df[gk_cols].mean(axis=1)
    else:
        df["goalkeeping"] = 0 
    
    return df

# --- 5. Encode play style ---
def encode_play_style(df):
    def split_styles(s):
        if pd.isna(s) or s == "None":
            return ["None"]
        return [style.strip().replace(" ", "_") for style in re.split(r',|;', s)]
    
    df["play_style_list"] = df["play_style"].apply(split_styles)
    
    unique_styles = sorted({style for sublist in df["play_style_list"] for style in sublist})
    for style in unique_styles:
        df[f"playstyle_{style}"] = df["play_style_list"].apply(lambda lst: 1 if style in lst else 0)
    
    df = df.drop("play_style_list", axis=1)
    return df

# --- 6. Encode technical skills and foot preference ---
def encode_skills_and_foot(df):
    df["preferred_foot_encoded"] = df["preferred_foot"].apply(lambda x: 1 if str(x).lower().startswith("r") else 0)
    
    # Weak foot and skill moves scaled to 0-1
    df["weak_foot_scaled"] = df["weak_foot"] / 5
    df["skill_moves_scaled"] = df["skill_moves"] / 5
    return df


df = create_all_positions(df)
df = encode_positions(df)
df = assign_position_group(df)
df = compute_composite_attributes(df)
df = encode_play_style(df)
df = encode_skills_and_foot(df)

df.to_csv("/Users/srikarpoladi/Downloads/Fifa25data/data/processed/fifa25_features.csv", index=False)
print("Feature engineering complete, dataset saved!")