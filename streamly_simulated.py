import random
import pandas as pd
from river import compose, preprocessing, tree, metrics, drift
import numbers

# 1. Generate realistic movie watch event stream
def generate_stream(num_samples=1000):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    genres = ["Action", "Comedy", "Romance", "Horror", "Drama", "Sci-Fi"]

    stream = []
    for i in range(num_samples):
        hour = random.randint(0, 23)
        day = random.choice(days)
        duration = random.randint(70, 180)

        # Context-aware genre preferences
        if hour in range(8, 12):  # Morning
            possible_genres = ["Comedy", "Drama"]
        elif hour in range(12, 18):  # Afternoon
            possible_genres = ["Action", "Sci-Fi", "Comedy"]
        elif hour in range(18, 23):  # Evening
            possible_genres = ["Romance", "Drama", "Comedy"]
        else:  # Late night
            possible_genres = ["Horror", "Thriller", "Sci-Fi"]

        # Simulate concept drift halfway through
        if i < num_samples * 0.6:
            genre = random.choices(
                possible_genres,
                weights=[3 if g in ["Comedy", "Action"] else 1 for g in possible_genres]
            )[0]
        else:
            genre = random.choices(
                possible_genres,
                weights=[3 if g in ["Romance", "Horror"] else 1 for g in possible_genres]
            )[0]

        stream.append({
            "duration": duration,
            "hour": hour,
            "day": day,
            "genre": genre
        })

    return stream

# 2. Define River pipeline
def get_pipeline():
    cat = (
        compose.SelectType(str)
        | preprocessing.StatImputer()
        | preprocessing.OneHotEncoder()
    )
    num = (
        compose.SelectType(numbers.Number)
        | preprocessing.StatImputer()
    )
    model = tree.HoeffdingTreeClassifier()
    return (num + cat) | model

# 3. Initialize
pipeline = get_pipeline()
adwin = drift.ADWIN()
f1 = metrics.MicroF1()
cm = metrics.ConfusionMatrix()

stream = generate_stream(100000)

# 4. Run stream and track results
records = []
drift_points = []

for i, event in enumerate(stream):
    x = event.copy()
    y = x.pop("genre")
    y_pred = pipeline.predict_one(x)

    correct = 0
    error = 1
    if y_pred is not None:
        correct = int(y == y_pred)
        error = int(y != y_pred)
        f1.update(y, y_pred)
        cm.update(y, y_pred)

    adwin.update(correct)
    drift_detected = adwin.drift_detected

    if drift_detected:
        print(f"⚠️ Drift detected at sample #{i} — True Genre: {y}, Predicted: {y_pred}")
        drift_points.append(i)
        adwin._reset()

    pipeline.learn_one(x, y)

    records.append({
        "Sample #": i,
        "Day": event["day"],
        "Hour": event["hour"],
        "Duration": event["duration"],
        "True Genre": y,
        "Predicted Genre": y_pred,
        "Correct?": correct,
        "Error?": error,
        "F1 Score": round(f1.get(), 3),
        "Drift Detected": drift_detected
    })

# 5. Display summary results
df = pd.DataFrame(records)
print(df.head(20))
print("\n🧠 Final F1 Score:", round(f1.get(), 3))
print("✅ Final Confusion Matrix:\n", cm)
print("\n📌 Drift Events:\n", df[df["Drift Detected"] == True])
