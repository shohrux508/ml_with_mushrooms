"""
train_pipeline.py — Полный ML-пайплайн:
  • Обучение Random Forest (финальная модель) + сохранение .pkl
  • Кластеризация (k-means, DBSCAN, Silhouette)
  • Ассоциативные правила (Apriori)
  • Поиск аномалий (Isolation Forest, LOF)
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "mushroom_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "ml", "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "ml", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Загрузка и предобработка
# ═════════════════════════════════════════════════════════════════════════════
def load_and_preprocess():
    """Загрузка данных, кодирование, разбиение."""
    df = pd.read_csv(DATA_PATH)

    # Удаляем бесполезный признак
    if "veil_type" in df.columns:
        df = df.drop(columns=["veil_type"])

    # Обработка пропусков
    df["stalk_root"] = df["stalk_root"].replace("?", "Unknown")

    # Целевая переменная
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Mushroom_quality"])  # e=0, p=1

    # Признаки
    X_cat = df.drop(columns=["Mushroom_quality"])
    feature_columns = list(X_cat.columns)
    X = pd.get_dummies(X_cat)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return df, X, y, X_train, X_test, y_train, y_test, le_target, feature_columns


# ═════════════════════════════════════════════════════════════════════════════
# 2. Обучение финальной модели + сохранение
# ═════════════════════════════════════════════════════════════════════════════
def train_and_save_model(X_train, y_train, X_test, y_test, X_columns, feature_columns):
    """Обучает Random Forest, сохраняет артефакты для inference."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight={0: 1, 1: 2},  # повышаем вес ядовитых
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    print("  Метрики Random Forest:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}")

    # Feature importance — top‑10
    importances = model.feature_importances_
    fi = pd.Series(importances, index=X_columns).sort_values(ascending=False)
    top_features = fi.head(10)

    # Сохранение графика
    plt.figure(figsize=(10, 5))
    top_features.plot(kind="barh", color="#8b5cf6")
    plt.gca().invert_yaxis()
    plt.title("Top-10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(PLOTS_DIR, "feature_importances.png")
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"  Feature importances сохранены → {fi_path}")

    # ── Сохраняем модель ──────────────────────────────────────────────────
    artifact = {
        "model": model,
        "ohe_columns": list(X_columns),
        "feature_columns": feature_columns,
        "metrics": metrics,
        "top_features": list(top_features.index),
    }

    model_path = os.path.join(MODEL_DIR, "mushroom_rf_model.pkl")
    joblib.dump(artifact, model_path)
    print(f"  Модель сохранена → {model_path}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# 3. Кластеризация (k-means, DBSCAN)
# ═════════════════════════════════════════════════════════════════════════════
def clustering_analysis(X):
    """k-means (k=2..6) + DBSCAN + Silhouette + PCA-визуализация."""
    print("\n--- Кластеризация ---")

    # PCA для визуализации
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # ── k-means ────────────────────────────────────────────────────────────
    silhouettes = {}
    best_k, best_sil = 2, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, sample_size=2000, random_state=42)
        silhouettes[k] = s
        print(f"  k-means k={k}  Silhouette={s:.4f}")
        if s > best_sil:
            best_sil = s
            best_k = k

    # График Silhouette
    plt.figure(figsize=(7, 4))
    plt.plot(list(silhouettes.keys()), list(silhouettes.values()), "o-", color="#06b6d4")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("k-means: Silhouette Score vs k")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "silhouette_kmeans.png"), bbox_inches="tight")
    plt.close()

    # PCA scatter лучший k-means
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km_best.fit_predict(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="viridis", s=5, alpha=0.6)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"k-means (k={best_k}) — PCA проекция")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "kmeans_pca.png"), bbox_inches="tight")
    plt.close()

    # ── DBSCAN ─────────────────────────────────────────────────────────────
    dbscan = DBSCAN(eps=5, min_samples=10)
    db_labels = dbscan.fit_predict(X)
    n_clusters_db = len(set(db_labels) - {-1})
    n_noise = (db_labels == -1).sum()
    print(f"  DBSCAN: кластеров={n_clusters_db}, шум={n_noise}")

    if n_clusters_db >= 2:
        mask = db_labels != -1
        db_sil = silhouette_score(X[mask], db_labels[mask], sample_size=2000, random_state=42)
        print(f"  DBSCAN Silhouette (без шума)={db_sil:.4f}")

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap="Spectral", s=5, alpha=0.6)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"DBSCAN — PCA проекция ({n_clusters_db} кластеров)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dbscan_pca.png"), bbox_inches="tight")
    plt.close()

    print("  Графики кластеризации сохранены.")
    return silhouettes, km_labels, db_labels


# ═════════════════════════════════════════════════════════════════════════════
# 4. Ассоциативные правила (Apriori)
# ═════════════════════════════════════════════════════════════════════════════
def association_rules_analysis(df):
    """Apriori на бинаризированных признаках."""
    print("\n--- Ассоциативные правила (Apriori) ---")

    # Подготовка «транзакций»: каждая строка = набор "признак=значение"
    transactions = []
    cols_to_use = [c for c in df.columns if c != "Mushroom_quality"]
    for _, row in df.iterrows():
        transactions.append([f"{col}={row[col]}" for col in cols_to_use])

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori
    freq_items = apriori(basket, min_support=0.3, use_colnames=True)
    print(f"  Частые наборы (support ≥ 0.3): {len(freq_items)}")

    if len(freq_items) > 0:
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.8)
        rules = rules.sort_values("lift", ascending=False)
        print(f"  Ассоциативные правила (confidence ≥ 0.8): {len(rules)}")

        # Сохранение top-20 правил
        top_rules = rules.head(20)[["antecedents", "consequents", "support", "confidence", "lift"]]
        top_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ", ".join(x))
        top_rules["consequents"] = top_rules["consequents"].apply(lambda x: ", ".join(x))
        rules_path = os.path.join(PLOTS_DIR, "association_rules_top20.csv")
        top_rules.to_csv(rules_path, index=False)
        print(f"  Top-20 правил сохранены → {rules_path}")

        # Визуализация
        if len(rules) > 0:
            plt.figure(figsize=(10, 6))
            plot_rules = rules.head(50)
            scatter = plt.scatter(
                plot_rules["support"],
                plot_rules["confidence"],
                c=plot_rules["lift"],
                cmap="YlOrRd",
                s=plot_rules["lift"] * 20,
                alpha=0.7,
                edgecolors="k",
                linewidths=0.3,
            )
            plt.colorbar(scatter, label="Lift")
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.title("Ассоциативные правила (Support vs Confidence, цвет=Lift)")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "apriori_rules.png"), bbox_inches="tight")
            plt.close()
            print("  График ассоциативных правил сохранён.")

        return rules
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 5. Поиск аномалий (Isolation Forest + LOF)
# ═════════════════════════════════════════════════════════════════════════════
def anomaly_detection(X, y):
    """Isolation Forest и LOF."""
    print("\n--- Поиск аномалий ---")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # ── Isolation Forest ───────────────────────────────────────────────────
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso.fit_predict(X)  # 1=normal, -1=anomaly

    n_anomaly_iso = (iso_pred == -1).sum()
    print(f"  Isolation Forest: аномалий={n_anomaly_iso} ({n_anomaly_iso/len(X)*100:.1f}%)")

    # Среди аномалий — сколько ядовитых?
    anomaly_mask = iso_pred == -1
    if anomaly_mask.sum() > 0:
        poison_in_anomaly = y[anomaly_mask].sum()
        print(
            f"  Из них ядовитых: {poison_in_anomaly} "
            f"({poison_in_anomaly/anomaly_mask.sum()*100:.1f}%)"
        )

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = np.where(iso_pred == -1, "red", "steelblue")
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=5, alpha=0.5)
    axes[0].set_title(f"Isolation Forest ({n_anomaly_iso} аномалий)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # ── LOF ────────────────────────────────────────────────────────────────
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_pred = lof.fit_predict(X)

    n_anomaly_lof = (lof_pred == -1).sum()
    print(f"  LOF: аномалий={n_anomaly_lof} ({n_anomaly_lof/len(X)*100:.1f}%)")

    colors_lof = np.where(lof_pred == -1, "red", "steelblue")
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors_lof, s=5, alpha=0.5)
    axes[1].set_title(f"LOF ({n_anomaly_lof} аномалий)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "anomaly_detection.png"), bbox_inches="tight")
    plt.close()
    print("  Графики аномалий сохранены.")

    return iso_pred, lof_pred


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  Mushroom ML Pipeline — полный запуск")
    print("=" * 60)

    # 1. Загрузка
    print("\n--- 1. Загрузка и предобработка ---")
    df, X, y, X_train, X_test, y_train, y_test, le_target, feature_columns = (
        load_and_preprocess()
    )
    print(f"  Размер: {df.shape}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Обучение и сохранение модели
    print("\n--- 2. Обучение Random Forest ---")
    model, metrics = train_and_save_model(
        X_train, y_train, X_test, y_test, X.columns, feature_columns
    )

    # 3. Кластеризация
    clustering_analysis(X)

    # 4. Ассоциативные правила
    association_rules_analysis(df)

    # 5. Поиск аномалий
    anomaly_detection(X, y)

    print("\n" + "=" * 60)
    print("  ✓ Пайплайн завершён!")
    print(f"  Модель: {os.path.join(MODEL_DIR, 'mushroom_rf_model.pkl')}")
    print(f"  Графики: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
