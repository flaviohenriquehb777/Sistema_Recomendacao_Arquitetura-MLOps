import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import PredictionErrorDisplay


def plot_coeficientes(df_coeficientes, titulo="Coeficientes"):
    df_coeficientes.plot.barh()

    plt.title(titulo)
    plt.axvline(x=0, color="0.5")
    plt.xlabel("Coeficientes")
    
    plt.gca().get_legend().remove()
    
    plt.show()


def plot_residuos(y_true, y_pred):
    residuos = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    h = sns.histplot(residuos, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        ax=axs[1],
    )
    
    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=axs[2],
    )

    plt.tight_layout()

    plt.show()


def plot_comparar_metricas_modelos(df_resultados):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()
