import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def load_and_eda(filepath: str) -> pd.DataFrame:
    """Шаг 1: Загрузка и EDA (Exploratory Data Analysis)"""
    print("--- Шаг 1: Загрузка и EDA ---")
    
    # Загружаем датасет
    df = pd.read_csv(filepath)
    
    # Выводим первые 5 строк
    print("\nПервые 5 строк датасета:")
    print(df.head())
    
    # Выводим общую информацию
    print("\nИнформация о датасете:")
    df.info()
    
    # График 1: Распределение целевой переменной (Mushroom_quality)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Mushroom_quality', palette='Set2')
    plt.title('Распределение целевой переменной (Mushroom_quality)')
    plt.xlabel('Съедобность (e=съедобный, p=ядовитый)')
    plt.ylabel('Количество')
    plt.savefig('distribution_mushroom_quality.png')
    plt.close()
    
    # График 2: Визуализация 2-3 ключевых признаков (запах - odor, цвет шляпки - cap_color)
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='odor', hue='Mushroom_quality', palette='Set1')
    plt.title('Распределение запаха (odor) по качеству гриба')
    plt.xlabel('Запах (odor)')
    plt.ylabel('Количество')
    plt.savefig('distribution_odor.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='cap_color', hue='Mushroom_quality', palette='Set1')
    plt.title('Распределение цвета шляпки (cap_color) по качеству гриба')
    plt.xlabel('Цвет шляпки (cap_color)')
    plt.ylabel('Количество')
    plt.savefig('distribution_cap_color.png')
    plt.close()
    
    print("\nГрафики для EDA сохранены в рабочую директорию.")
    return df

def preprocess_data(df: pd.DataFrame):
    """Шаг 2: Предобработка данных"""
    print("\n--- Шаг 2: Предобработка данных ---")
    
    # 1. Очистка: удаляем константные признаки
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        print(f"Удаление константных признаков: {const_cols}")
        df = df.drop(columns=const_cols)
        
    # 2. Обработка пропусков: если есть 'stalk_root', заменяем '?' на 'Unknown'
    if 'stalk_root' in df.columns:
        print("Замена пропусков '?' в stalk_root на 'Unknown'.")
        df['stalk_root'] = df['stalk_root'].replace('?', 'Unknown')
        
    # 3. Кодирование: LabelEncoder для целевой переменной (e=0, p=1)
    print("Кодирование целевой переменной Mushroom_quality: e=0 (съедобный), p=1 (ядовитый).")
    le = LabelEncoder()
    df['Mushroom_quality'] = le.fit_transform(df['Mushroom_quality'])
    # Примечание: le.classes_ покажет ['e', 'p']. Таким образом, 'e' -> 0, 'p' -> 1.
    
    X = df.drop(columns=['Mushroom_quality'])
    y = df['Mushroom_quality']
    
    # 4. Кодирование: Применяем get_dummies для всех остальных признаков (One-Hot Encoding)
    print("Применение get_dummies (One-Hot Encoding) для категориальных признаков.")
    X = pd.get_dummies(X, drop_first=True)
    
    # 5. Разбиение на train и test в пропорции 70/30 с stratify
    print("Разбиение выборки на train и test (70/30) со стратификацией.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Шаг 3: Обучение моделей"""
    print("\n--- Шаг 3: Обучение моделей ---")
    
    print("Обучение Gaussian Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    print("Обучение RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return {'Naive Bayes': nb_model, 'Random Forest': rf_model}

def evaluate_and_compare(models: dict, X_test, y_test):
    """Шаг 4: Оценка и сравнение"""
    print("\n--- Шаг 4: Оценка и сравнение ---")
    
    results = []
    
    # Подготавливаем фигуру для ROC-кривых
    plt.figure(1, figsize=(8, 6))
    
    for name, model in models.items():
        print(f"\nМодель: {name}")
        y_pred = model.predict(X_test)
        
        # Получение вероятностей для ROC-AUC. 
        # GaussianNB и RandomForestClassifier поддерживают predict_proba.
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 1. Считаем метрики. Нас интересуют Precision и Recall специально для ядовитых грибов (класс 1).
        # pos_label=1 по умолчанию, так что metric_score считает их для p=1.
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f} (для ядовитых)")
        print(f"  Recall:    {rec:.4f} (для ядовитых)")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        
        # 2. Матрица ошибок (отдельный график)
        cm = confusion_matrix(y_test, y_pred)
        print("  Confusion Matrix:")
        print(f"  {cm[0]}\n  {cm[1]}")
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Edible (0)', 'Poisonous (1)'], 
                    yticklabels=['Edible (0)', 'Poisonous (1)'], ax=ax)
        ax.set_title(f'Матрица ошибок - {name}')
        ax.set_xlabel('Предсказано')
        ax.set_ylabel('Истина')
        fig.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close(fig)
        
        # 3. Данные для ROC-кривой (добавляем график на общую фигуру)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(1) # Убеждаемся, что мы на первой фигуре
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})')
        
        # Сохранение результатов для таблицы
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Poisonous)': prec,
            'Recall (Poisonous)': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
    # Доработка и сохранение ROC-кривых
    plt.figure(1)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Кривые (ROC Curves)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()
    
    # 4. Итоговая таблица со сравнением метрик
    print("\n--- Итоговое сравнение моделей ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

def main():
    filepath = 'mushroom_data.csv'
    
    # Шаг 1: Загрузка + EDA
    df = load_and_eda(filepath)
    
    # Шаг 2: Предобработка
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Шаг 3: Обучение
    models = train_models(X_train, y_train)
    
    # Шаг 4: Оценка и сравнение
    evaluate_and_compare(models, X_test, y_test)

if __name__ == "__main__":
    main()
