# 🎬 Streamly: Real-Time Genre Prediction with Drift Detection using River

Welcome to **Streamly**, a simulated movie streaming service that demonstrates how to build a **real-time, continuously learning machine learning system** using the [River](https://riverml.xyz) library.

This project showcases:
- 🔄 Real-time genre prediction
- 🧠 Online learning (no batch retraining)
- 📉 Concept drift detection using ADWIN
- ✅ Streaming metrics (F1, confusion matrix)
- 📊 Live performance monitoring via terminal

---

## 🚀 Problem Overview

**Streamly** aims to predict which movie genre a user is most likely to watch next — based on features like:
- `hour` of the day
- `day` of the week
- `duration` of the last watched movie

But user preferences change! What worked yesterday may fail today. That’s where **online learning and concept drift detection** come in.

---

## 🧪 Real-Time ML with River

This project uses:

- 🧱 `river.compose`, `preprocessing`, `tree` – to build a real-time learning pipeline
- 🔍 `river.metrics` – to monitor F1 and confusion matrix over time
- ⚠️ `river.drift.ADWIN` – to detect concept drift in predictions
- 📋 `pandas` – to log predictions, errors, and drifts for analysis

---

## 🔁 Example Output (Terminal)

```
⚠️ Drift detected at sample #63391 – True Genre: Sci-Fi, Predicted: Action

Sample #  | True Genre | Predicted | F1 Score | Drift
--------- | -----------|-----------|----------|-------
0         | Comedy     | None      | 0.000    | False
1         | Comedy     | Sci-Fi    | 0.000    | False
...
63391     | Sci-Fi     | Action    | 0.472    | ✅ Drift
```

---

## 🧠 What Is Concept Drift?

> Concept drift occurs when the relationship between input features and target labels changes over time.

For example, if a user used to watch **Comedy on Fridays**, but now prefers **Drama**, the model’s assumptions become outdated. Streamly uses River's **ADWIN** to detect this and respond.

---

## 📦 Installation

```bash
pip install river pandas matplotlib
```

Make sure you're using `river>=0.22.0`.

---

## 🛠️ How to Run

```bash
python streamly_simulated_drift.py
```

You’ll see:
- Live genre predictions
- Model performance (F1 score)
- When and why drift was detected
- A confusion matrix at the end
- A drift log table printed to terminal

---

## 📁 Project Structure

```
├── streamly_simulated_drift.py     # Main script with simulation + drift detection
├── README.md                       # Project overview
```

---

## ✅ Highlights

- No batch training needed
- Online preprocessing and model updates
- Drift detection using prediction history
- Lightweight, explainable, and production-ready approach

---

## 🧠 Want to Learn More?

- [River documentation](https://riverml.xyz)
- [Concept drift explanation](https://en.wikipedia.org/wiki/Concept_drift)
- [Why streaming ML matters](https://huyenchip.com/2022/02/07/data-centric-ai.html)

---

## ✨ Author

Built as part of a project for **AI Engineering / MLOps coursework**.  
Maintained by Huating Sun.

---

## 📝 License

This project is licensed under the MIT License.
