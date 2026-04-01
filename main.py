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

def main():
    print("--- Шаг 1: Загрузка и EDA ---")
    # Загрузка данных
    df = pd.read_csv('mushroom_data.csv')
    
    # Вывод первых 5 строк
    print("Первые 5 строк датасета:")
    print(df.head())
    
    # Общая информация о датасете
    print("\nИнформация о датасете:")
    df.info()
    
    # Визуализация баланса классов (целевая переменная)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Mushroom_quality', data=df, palette='Set2')
    plt.title('Распределение целевой переменной (Mushroom_quality)')
    plt.xlabel('Качество (e=съедобный, p=ядовитый)')
    plt.ylabel('Количество')
    plt.savefig('target_distribution.png')
    plt.close()
    
    # Визуализация распределения признаков odor и cap_color
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
    plt.savefig('features_distribution.png')
    plt.close()
    
    print("\nГрафики EDA сохранены: target_distribution.png, features_distribution.png")

    print("\n--- Шаг 2: Предобработка данных ---")
    # Удаление константного признака veil_type (не влияет на результат)
    if 'veil_type' in df.columns:
        df = df.drop(columns=['veil_type'])
    
    # Обработка пропусков: замена '?' на 'Unknown' в stalk_root
    df['stalk_root'] = df['stalk_root'].replace('?', 'Unknown')
    
    # Кодирование целевой переменной: e=0 (съедобный), p=1 (ядовитый)
    le = LabelEncoder()
    # Целевая переменная: Mushroom_quality
    # LabelEncoder сортирует по алфавиту: 'e' будет 0, 'p' будет 1
    y = le.fit_transform(df['Mushroom_quality'])
    
    # Кодирование категориальных признаков c помощью One-Hot Encoding
    X_cat = df.drop(columns=['Mushroom_quality'])
    X = pd.get_dummies(X_cat) # Не drop_first, так как для Random Forest/Naive Bayes это не страшно, но можно и drop_first=True
    
    # Разбиение на train и test (70/30, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    print("\n--- Шаг 3: Обучение моделей ---")
    # 1. Наивный Байес
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    # 2. Случайный лес
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    print("Модели успешно обучены.")

    print("\n--- Шаг 4: Оценка и сравнение ---")
    
    models = {
        'Naive Bayes': nb_model,
        'Random Forest': rf_model
    }
    
    results = []

    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        # Предсказания
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Метрики. Мы делаем упор на ядовитые грибы (которые закодированы как 1)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Poisonous)': prec,
            'Recall (Poisonous)': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Edible (0)', 'Poisonous (1)'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Матрица ошибок - {name}')
        fig_cm.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close(fig_cm)
        
        # Данные для ROC-кривой
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    # Завершение графика ROC-кривой
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    print("Матрицы ошибок и ROC-кривая сохранены.")
    
    # Сравнение
    results_df = pd.DataFrame(results)
    print("\nИтоговая таблица с метриками обеих моделей:")
    print("-" * 80)
    print(results_df.to_string(index=False))
    print("-" * 80)

if __name__ == "__main__":
    main()
