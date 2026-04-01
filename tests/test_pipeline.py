"""
tests/test_pipeline.py — Тесты для ML пайплайна классификации грибов.
Запуск: pytest tests/ -v
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Добавляем корень проекта в sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "ml"))


# ═════════════════════════════════════════════════════════════════════════════
# Фикстуры
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_df():
    """Загрузка датасета."""
    csv_path = os.path.join(ROOT_DIR, "mushroom_data.csv")
    assert os.path.exists(csv_path), f"Датасет не найден: {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def pipeline_data(raw_df):
    """Полная предобработка и обучение (один раз на все тесты модуля)."""
    from main import step2_preprocess, step3_train_models

    preprocessed = step2_preprocess(raw_df)
    models = step3_train_models(preprocessed["X_train"], preprocessed["y_train"])
    return {**preprocessed, "models": models}


# ═════════════════════════════════════════════════════════════════════════════
# Тесты загрузки данных
# ═════════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    """Тесты загрузки и структуры данных."""

    def test_csv_exists(self):
        csv_path = os.path.join(ROOT_DIR, "mushroom_data.csv")
        assert os.path.exists(csv_path)

    def test_shape(self, raw_df):
        """Датасет должен содержать 8124 строки и 23 столбца."""
        assert raw_df.shape[0] > 0, "Датасет пуст"
        assert raw_df.shape[1] >= 22, f"Ожидалось >= 22 столбцов, получили {raw_df.shape[1]}"

    def test_target_column_exists(self, raw_df):
        assert "Mushroom_quality" in raw_df.columns

    def test_target_values(self, raw_df):
        """Целевая переменная должна содержать только 'e' и 'p'."""
        unique = set(raw_df["Mushroom_quality"].unique())
        assert unique == {"e", "p"}, f"Неожиданные значения: {unique}"


# ═════════════════════════════════════════════════════════════════════════════
# Тесты предобработки
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    """Тесты предобработки данных."""

    def test_train_test_split_not_empty(self, pipeline_data):
        assert pipeline_data["X_train"].shape[0] > 0
        assert pipeline_data["X_test"].shape[0] > 0

    def test_train_test_proportions(self, pipeline_data):
        """Test/train split ≈ 70/30."""
        total = pipeline_data["X_train"].shape[0] + pipeline_data["X_test"].shape[0]
        test_ratio = pipeline_data["X_test"].shape[0] / total
        assert 0.25 <= test_ratio <= 0.35, f"Ожидалась доля теста ~0.3, получили {test_ratio:.2f}"

    def test_no_nulls_after_preprocessing(self, pipeline_data):
        """После предобработки не должно быть NaN."""
        assert not pipeline_data["X_train"].isnull().any().any()
        assert not pipeline_data["X_test"].isnull().any().any()

    def test_one_hot_encoding_shape(self, pipeline_data):
        """One-Hot Encoding должен расширить признаки."""
        assert pipeline_data["X_train"].shape[1] > 22, (
            f"После OHE ожидалось > 22 признаков, получили {pipeline_data['X_train'].shape[1]}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Тесты моделей
# ═════════════════════════════════════════════════════════════════════════════

class TestModels:
    """Тесты обучения и качества моделей."""

    def test_models_trained(self, pipeline_data):
        """Оба алгоритма должны быть обучены."""
        assert "Naive Bayes" in pipeline_data["models"]
        assert "Random Forest" in pipeline_data["models"]

    def test_naive_bayes_accuracy(self, pipeline_data):
        """Naive Bayes: точность > 80%."""
        from sklearn.metrics import accuracy_score

        model = pipeline_data["models"]["Naive Bayes"]
        y_pred = model.predict(pipeline_data["X_test"])
        acc = accuracy_score(pipeline_data["y_test"], y_pred)
        assert acc > 0.80, f"NB Accuracy={acc:.4f} < 0.80"

    def test_random_forest_accuracy(self, pipeline_data):
        """Random Forest: точность > 95%."""
        from sklearn.metrics import accuracy_score

        model = pipeline_data["models"]["Random Forest"]
        y_pred = model.predict(pipeline_data["X_test"])
        acc = accuracy_score(pipeline_data["y_test"], y_pred)
        assert acc > 0.95, f"RF Accuracy={acc:.4f} < 0.95"

    def test_random_forest_recall_poisonous(self, pipeline_data):
        """Recall для ядовитых (класс 1) должен быть > 95%
        (в контексте безопасности критически важен recall)."""
        from sklearn.metrics import recall_score

        model = pipeline_data["models"]["Random Forest"]
        y_pred = model.predict(pipeline_data["X_test"])
        recall = recall_score(pipeline_data["y_test"], y_pred)
        assert recall > 0.95, (
            f"RF Recall(poisonous)={recall:.4f} < 0.95 — "
            "пропускаем слишком много ядовитых!"
        )

    def test_random_forest_predict_proba(self, pipeline_data):
        """Модель должна возвращать вероятности."""
        model = pipeline_data["models"]["Random Forest"]
        proba = model.predict_proba(pipeline_data["X_test"][:5])
        assert proba.shape == (5, 2), f"Ожидалось shape (5,2), получили {proba.shape}"
        # Вероятности суммируются в 1
        sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(5), decimal=5)


# ═════════════════════════════════════════════════════════════════════════════
# Тесты train_pipeline.py
# ═════════════════════════════════════════════════════════════════════════════

class TestTrainPipeline:
    """Тесты для скрипта ml/train_pipeline.py."""

    def test_load_and_preprocess(self):
        from train_pipeline import load_and_preprocess

        df, X, y, X_train, X_test, y_train, y_test, le_target, feature_columns = (
            load_and_preprocess()
        )
        assert df.shape[0] > 0
        assert len(feature_columns) > 0
        assert len(y) == df.shape[0]
        # Проверка стратификации
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.02, (
            "Стратификация нарушена: train и test имеют разное распределение классов"
        )

    def test_model_save_load(self, tmp_path):
        """Проверяем, что модель сохраняется и загружается корректно."""
        import joblib
        from train_pipeline import load_and_preprocess

        df, X, y, X_train, X_test, y_train, y_test, le_target, feature_columns = (
            load_and_preprocess()
        )

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        artifact = {
            "model": model,
            "ohe_columns": list(X.columns),
            "feature_columns": feature_columns,
        }

        path = tmp_path / "test_model.pkl"
        joblib.dump(artifact, path)
        loaded = joblib.load(path)

        assert "model" in loaded
        assert "ohe_columns" in loaded
        y_pred = loaded["model"].predict(X_test[:5])
        assert len(y_pred) == 5


# ═════════════════════════════════════════════════════════════════════════════
# Тесты API (unit-level)
# ═════════════════════════════════════════════════════════════════════════════

class TestAPI:
    """Базовые тесты для API endpoints."""

    def test_feature_options_completeness(self):
        """Все признаки из датасета должны быть описаны в FEATURE_OPTIONS."""
        sys.path.insert(0, ROOT_DIR)
        from api import FEATURE_OPTIONS

        expected_features = [
            "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
            "gill_attachment", "gill_spacing", "gill_size", "gill_color",
            "stalkshape", "stalk_root",
            "stalk_surface_above_ring", "stalk_surface_below_ring",
            "stalk_color_above_ring", "stalk_color_below_ring",
            "veil_color", "ring_number", "ring_type",
            "spore_print_color", "population", "habitat",
        ]

        for feat in expected_features:
            assert feat in FEATURE_OPTIONS, f"Признак '{feat}' отсутствует в FEATURE_OPTIONS"
            assert "label" in FEATURE_OPTIONS[feat]
            assert "options" in FEATURE_OPTIONS[feat]
            assert len(FEATURE_OPTIONS[feat]["options"]) > 0

    def test_top_features_valid(self):
        """TOP_FEATURES должны быть подмножеством FEATURE_OPTIONS."""
        from api import FEATURE_OPTIONS, TOP_FEATURES

        for feat in TOP_FEATURES:
            assert feat in FEATURE_OPTIONS, f"Top feature '{feat}' не найден в FEATURE_OPTIONS"
