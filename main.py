from flask import Flask, jsonify, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask Application
app = Flask(__name__)

# 1. Load pre-trained dataset and TF-IDF vectorizer via Pickle
try:
  with open("course_compass_df.pkl", "rb") as f:
    df = pickle.load(f)

  with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

  # Pre-calculate the TF-IDF feature matrix on server startup
  tfidf_matrix = tfidf.transform(df["combined_features"])
  print("SUCCESS: Model and dataset loaded successfully.")
except Exception as e:
  print(f"ERROR: Could not load model files. Details: {e}")


# 2. Serve the Frontend HTML page
@app.route("/", methods=["GET"])
def home():
  return render_template("index.html")


# 3. Recommendation API Endpoint (POST)
@app.route("/recommend", methods=["POST"])
def recommend_courses():
  data = request.get_json()

  if not data or "student_stream" not in data or "student_interests" not in data:
    return (
        jsonify({
            "status": "error",
            "message": (
                "Missing required fields: 'student_stream' and"
                " 'student_interests'"
            ),
        }),
        400,
    )

  student_stream = data["student_stream"]
  student_interests = data["student_interests"]
  top_n = int(data.get("top_n", 5))

  # Rule-based filtering (Stream match or 'ANY')
  eligible_df = df[
      df["eligible_streams"].str.contains(
          student_stream, case=False, na=False
      )
      | df["eligible_streams"].str.contains("ANY", case=False, na=False)
  ].copy()

  if eligible_df.empty:
    return jsonify({
        "status": "error",
        "message": f"No eligible courses found for stream: {student_stream}",
        "recommendations": [],
    })

  # Cosine similarity ranking
  eligible_indices = eligible_df.index.tolist()
  query_vec = tfidf.transform([student_interests])
  subset_matrix = tfidf_matrix[eligible_indices]

  similarity_scores = cosine_similarity(query_vec, subset_matrix).flatten()
  eligible_df["match_score"] = similarity_scores

  # Sort and select top N results
  ranked_df = eligible_df.sort_values(by="match_score", ascending=False).head(
      top_n
  )

  results = []
  for _, row in ranked_df.iterrows():
    results.append({
        "course_id": row["course_id"],
        "course_name": row["course_name"],
        "degree_level": row["degree_level"],
        "duration_years": row["duration_years"],
        "career_paths": row["career_paths"],
        "match_score": round(float(row["match_score"]) * 100, 2),
    })

  return jsonify({
      "status": "success",
      "stream": student_stream,
      "query": student_interests,
      "count": len(results),
      "recommendations": results,
  })


if __name__ == "__main__":
  app.run(debug=True, port=5000)