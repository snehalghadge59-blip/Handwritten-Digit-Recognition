"""
train_model.py
──────────────
Trains a KNN classifier on sklearn's built-in digits dataset (8x8 images,
upscaled to 28x28 for consistency with user drawing canvas).
Applies GridSearchCV + K-Fold cross-validation exactly as described in resume.
"""

import time
import numpy as np
import pickle
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy.ndimage import zoom

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "knn_mnist.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

def upscale_to_28x28(X_8x8):
    """Upscale 8x8 digit images to 28x28 via bilinear zoom."""
    result = []
    for row in X_8x8:
        img = row.reshape(8, 8)
        img_28 = zoom(img, 28/8, order=1)  # bilinear
        result.append(img_28.flatten())
    return np.array(result)

def train():
    print("📥 Loading sklearn digits dataset (8×8 → upscaled to 28×28)...")
    digits = load_digits()
    X_raw, y = digits.data, digits.target  # X_raw shape: (1797, 64)

    # Upscale to 28×28 so canvas drawings (also 28×28) match training space
    X = upscale_to_28x28(X_raw)  # (1797, 784)
    print(f"   Samples: {len(X)}, Features: {X.shape[1]}")

    # Normalize pixel values to [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Data ready — Train: {len(X_train)}, Test: {len(X_test)}")

    # GridSearchCV for best k (as per resume)
    print("🔍 Running GridSearchCV for best hyperparameters...")
    param_grid = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
    grid = GridSearchCV(
        KNeighborsClassifier(algorithm="ball_tree", n_jobs=-1),
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=0
    )
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    print(f"   Best params: {best_params}")

    # Final model with best params
    knn = grid.best_estimator_

    # K-Fold cross-validation (as per resume)
    print("📊 K-Fold cross-validation (5-fold)...")
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"   CV Scores: {cv_scores.round(4)}")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final evaluation
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Model saved to {MODEL_PATH}")
    print(f"💾 Scaler saved to {SCALER_PATH}")
    return acc

if __name__ == "__main__":
    t0 = time.time()
    acc = train()
    print(f"\n⏱  Total training time: {time.time()-t0:.1f}s")
