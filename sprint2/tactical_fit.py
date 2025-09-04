import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def define_playstyle_profile(profile_dict):
    """
    Example profile_dict:
    {"attack": 0.9, "passing": 0.8, "dribbling": 0.85, "defense": 0.5, "physical": 0.7}
    """
    max_val = max(profile_dict.values())
    if max_val > 1:
        profile_dict = {k: v / max_val for k, v in profile_dict.items()}
    return profile_dict

def compute_player_fit(df, profile, fit_weight=0.7, quality_weight=0.3):
    """
    Computes both tactical fit (cosine similarity) and combined fit + quality score.
    """
    features = list(profile.keys())
    profile_vector = np.array([profile[f] for f in features]).reshape(1, -1)
    player_vectors = df[features].values

    # Tactical fit (cosine similarity)
    similarity = cosine_similarity(player_vectors, profile_vector)
    df["fit_score"] = similarity.flatten()

    # Overall quality = average of attributes (scaled 0–1 if needed)
    df["overall_score"] = df[features].mean(axis=1)

    # Weighted combination (default: 70% fit, 30% quality)
    df["combined_score"] = (
        fit_weight * df["fit_score"] + quality_weight * (df["overall_score"] / df["overall_score"].max())
    )

    return df

def rank_players_by_fit(df, top_n=10, use_combined=True):
    """
    Ranks players either by combined_score (default) or pure fit_score.
    """
    sort_col = "combined_score" if use_combined else "fit_score"
    return df.sort_values(sort_col, ascending=False).head(top_n)
