import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def plot_model_performance(models, mse_values, r2_values):
    """
    Exibe gráficos comparando o desempenho dos modelos com base no MSE e R².
    """
    model_names = list(models.keys())

    # Gráfico de barras para MSE
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, mse_values, color='skyblue')
    plt.title('Comparação de Modelos - Mean Squared Error (MSE)', fontsize=14)
    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.xticks(rotation=45)
    for i, v in enumerate(mse_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Gráfico de barras para R²
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, r2_values, color='lightgreen')
    plt.title('Comparação de Modelos - R² Score', fontsize=14)
    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.xticks(rotation=45)
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, predictions, model_names):
    """
    Exibe gráficos de dispersão para comparar valores reais vs. previstos.
    """
    plt.figure(figsize=(12, 8))
    for i, y_pred in enumerate(predictions):
        plt.scatter(y_test, y_pred, alpha=0.7, label=f"{model_names[i]} (R²={r2_score(y_test, y_pred):.2f})")

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal', linewidth=2)
    plt.title('Comparação: Valores Reais vs. Previstos', fontsize=14)
    plt.xlabel('Valores Reais (y_test)', fontsize=12)
    plt.ylabel('Valores Previstos (y_pred)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances):
    """
    Exibe a importância das features (apenas para modelos como Random Forest).
    """
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='purple')
    plt.title('Importância das Features', fontsize=14)
    plt.xlabel('Importância', fontsize=12)
    plt.gca().invert_yaxis()
    for i, v in enumerate(sorted_importances):
        plt.text(v + 0.01, i, f"{v:.2f}", fontsize=10)
    plt.tight_layout()
    plt.show()
