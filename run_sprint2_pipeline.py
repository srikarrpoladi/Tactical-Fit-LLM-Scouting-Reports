import pandas as pd
from sprint2 import clustering, tactical_fit, llm_scouting, visualize_clusters
from sprint2.tactical_profiles import TACTICAL_PROFILES

df = pd.read_csv("data/processed/fifa25_features.csv")

# Prepare features for clustering
X_scaled = clustering.prepare_clustering_features(df)

# Run clustering
labels, kmeans_model = clustering.run_kmeans(X_scaled, n_clusters=5)
df = clustering.add_cluster_labels(df, labels)
clustering.evaluate_clusters(X_scaled, labels)
df.to_csv("data/df_clustering_ready.csv", index=False)
print("Clustering complete and saved.")

# Tactical fit: compute top players
all_top_players = {}
for role_name, profile in TACTICAL_PROFILES.items():
    df_profile = tactical_fit.compute_player_fit(df.copy(), profile)
    top_players = tactical_fit.rank_players_by_fit(df_profile, top_n=5, use_combined=True)
    all_top_players[role_name] = top_players[["name", "fit_score", "overall_score", "combined_score"]]
    print(f"\n🏆 Top 5 players for {role_name.replace('_',' ').title()}:")
    print(all_top_players[role_name])

# Fine-tune tiny-GPT2 on full player stats
fine_tuned_dir = "fine_tuned_tiny_gpt2"
model, tokenizer = llm_scouting.fine_tune_model(df, output_dir=fine_tuned_dir, epochs=1)

# Generate scouting reports using fine-tuned model
reports_df = llm_scouting.generate_batch_reports(df, fine_tuned_dir=fine_tuned_dir, limit=10)
reports_df.to_csv("data/scouting_reports.csv", index=False)
print("Scouting reports saved to data/scouting_reports.csv")

# Visualizations
features_for_viz = ["attack","passing","dribbling","defense","physical","goalkeeping"]
visualize_clusters.plot_pca_clusters(df, features_for_viz)
visualize_clusters.plot_radar_for_cluster(df, cluster_label=0, features=features_for_viz)
