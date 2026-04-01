import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд (без Tkinter)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR = "."  # Директория для сохранения графиков


def step1_load_and_eda(plots_dir: str = PLOTS_DIR) -> dict:
    """Шаг 1: Загрузка данных и EDA. Возвращает df и пути к графикам."""
    df = pd.read_csv('mushroom_data.csv')

    # Визуализация баланса классов
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Mushroom_quality', data=df, palette='Set2')
    plt.title('Распределение целевой переменной (Mushroom_quality)')
    plt.xlabel('Качество (e=съедобный, p=ядовитый)')
    plt.ylabel('Количество')
    target_path = f'{plots_dir}/target_distribution.png'
    plt.savefig(target_path, bbox_inches='tight')
    plt.close()

    # Визуализация признаков odor и cap_color
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x='odor', hue='Mushroom_quality', data=df, ax=axes[0], palette='Set2')
    axes[0].set_title('Распределение запаха (odor) по классам')
    axes[0].set_xlabel('Запах')
    axes[0].set_ylabel('Количество')

    sns.countplot(x='cap_color', hue='Mushroom_quality', data=df, ax=axes[1], palette='Set2')
    axes[1].set_title('Распределение цвета шляпки (cap_color) по классам')
    axes[1].set_xlabel('Цвет шляпки')
    axes[1].set_ylabel('Количество')

    plt.tight_layout()
    features_path = f'{plots_dir}/features_distribution.png'
    plt.savefig(features_path, bbox_inches='tight')
    plt.close()

    return {
        "df": df,
        "shape": df.shape,
        "columns": list(df.columns),
        "plots": {
            "target_distribution": target_path,
            "features_distribution": features_path,
        }
    }


def step2_preprocess(df: pd.DataFrame) -> dict:
    """Шаг 2: Предобработка данных. Возвращает train/test выборки."""
    if 'veil_type' in df.columns:
        df = df.drop(columns=['veil_type'])

    df['stalk_root'] = df['stalk_root'].replace('?', 'Unknown')

    le = LabelEncoder()
    y = le.fit_transform(df['Mushroom_quality'])

    X_cat = df.drop(columns=['Mushroom_quality'])
    X = pd.get_dummies(X_cat)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }


def step3_train_models(X_train, y_train) -> dict:
    """Шаг 3: Обучение моделей. Возвращает обученные модели."""
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    return {
        "Naive Bayes": nb_model,
        "Random Forest": rf_model,
    }


def step4_evaluate(models: dict, X_test, y_test, plots_dir: str = PLOTS_DIR) -> dict:
    """Шаг 4: Оценка и сравнение моделей. Возвращает метрики и пути к графикам."""
    results = []
    plot_paths = {}

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        results.append({
            'Model': name,
            'Accuracy': round(acc, 6),
            'Precision (Poisonous)': round(prec, 6),
            'Recall (Poisonous)': round(rec, 6),
            'F1-Score': round(f1, 6),
            'ROC-AUC': round(auc, 6),
        })

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Edible (0)', 'Poisonous (1)'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Матрица ошибок - {name}')
        cm_path = f'{plots_dir}/confusion_matrix_{name.replace(" ", "_").lower()}.png'
        fig_cm.savefig(cm_path, bbox_inches='tight')
        plt.close(fig_cm)
        plot_paths[f'confusion_matrix_{name.lower().replace(" ", "_")}'] = cm_path

        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    roc_path = f'{plots_dir}/roc_curve.png'
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    plot_paths['roc_curve'] = roc_path

    return {
        "results": results,
        "plots": plot_paths,
    }


def run_full_pipeline(plots_dir: str = PLOTS_DIR) -> dict:
    """Запуск полного пайплайна. Возвращает все результаты."""
    eda = step1_load_and_eda(plots_dir)
    preprocessed = step2_preprocess(eda["df"])
    models = step3_train_models(preprocessed["X_train"], preprocessed["y_train"])
    evaluation = step4_evaluate(
        models,
        preprocessed["X_test"],
        preprocessed["y_test"],
        plots_dir,
    )

    return {
        "eda": {
            "shape": eda["shape"],
            "columns": eda["columns"],
            "plots": eda["plots"],
        },
        "preprocessing": {
            "train_shape": preprocessed["train_shape"],
            "test_shape": preprocessed["test_shape"],
        },
        "evaluation": evaluation,
    }


# ─── Запуск из терминала ───────────────────────────────────────────────────────
def main():
    print("--- Шаг 1: Загрузка и EDA ---")
    eda = step1_load_and_eda()
    print(f"  Датасет: {eda['shape'][0]} строк, {eda['shape'][1]} столбцов")
    print(f"  Графики сохранены: {', '.join(eda['plots'].values())}")

    print("\n--- Шаг 2: Предобработка данных ---")
    preprocessed = step2_preprocess(eda["df"])
    print(f"  Размер обучающей выборки: {preprocessed['train_shape']}")
    print(f"  Размер тестовой выборки:  {preprocessed['test_shape']}")

    print("\n--- Шаг 3: Обучение моделей ---")
    models = step3_train_models(preprocessed["X_train"], preprocessed["y_train"])
    print("  Модели успешно обучены.")

    print("\n--- Шаг 4: Оценка и сравнение ---")
    evaluation = step4_evaluate(
        models,
        preprocessed["X_test"],
        preprocessed["y_test"],
    )
    print("  Матрицы ошибок и ROC-кривая сохранены.")

    results_df = pd.DataFrame(evaluation["results"])
    print("\nИтоговая таблица с метриками обеих моделей:")
    print("-" * 80)
    print(results_df.to_string(index=False))
    print("-" * 80)


if __name__ == "__main__":
    main()
