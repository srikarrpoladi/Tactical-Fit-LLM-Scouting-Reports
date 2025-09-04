Tactical Fit & LLM Scouting Reports

How to Run

Install dependencies:

pip install -r requirements.txt


Run the pipeline:

python run_sprint2_pipeline.py

This project combines machine learning and LLMs to evaluate soccer players and generate automated scouting reports. Player data is clustered using KMeans (Silhouette score: 0.257) to identify tactical profiles, and top players are ranked for predefined roles.

A fine-tuned tiny-GPT2 generates concise scouting reports including strengths, weaknesses, stats, and play styles. The end-to-end pipeline handles data preparation, clustering, tactical fit evaluation, LLM report generation, and visualization