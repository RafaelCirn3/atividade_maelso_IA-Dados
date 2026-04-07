from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_ID = "rodrigoriboldi/incidentes-de-segurana-da-informao-no-brasil"
TARGET_COLUMN = "Ano"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def load_dataset() -> pd.DataFrame:
	dataset_directory = kagglehub.dataset_download(DATASET_ID)
	dataset_file = Path(dataset_directory) / "cert_2010-2019.csv"
	return pd.read_csv(dataset_file, sep=";")


def build_preprocessor() -> ColumnTransformer:
	numeric_features = ["Total", "Worm", "DOS", "Invasao", "Web", "Scan", "Fraude", "Outros"]
	categorical_features = ["Mes"]

	return ColumnTransformer(
		transformers=[
			("numeric", StandardScaler(), numeric_features),
			("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
		]
	)


def evaluate_models(feature_frame: pd.DataFrame, target_series: pd.Series) -> pd.DataFrame:
	train_features, test_features, train_target, test_target = train_test_split(
		feature_frame,
		target_series,
		test_size=0.2,
		random_state=42,
		stratify=target_series,
	)

	configurations = {
		"Euclidiana": {"metric": "euclidean"},
		"Manhattan": {"metric": "manhattan"},
		"Chebyshev": {"metric": "chebyshev"},
		"Minkowski": {"metric": "minkowski", "p": 3},
	}

	preprocessor = build_preprocessor()
	results = []

	for neighbors in range(1, 16):
		row = {"K": neighbors}
		for label, parameters in configurations.items():
			classifier = KNeighborsClassifier(n_neighbors=neighbors, **parameters)
			model = Pipeline(
				steps=[
					("preprocessor", preprocessor),
					("classifier", classifier),
				]
			)
			model.fit(train_features, train_target)
			predicted_target = model.predict(test_features)
			row[label] = accuracy_score(test_target, predicted_target)
		results.append(row)

	return pd.DataFrame(results)


def save_results_table(results_frame: pd.DataFrame) -> Path:
	output_path = OUTPUT_DIRECTORY / "knn_resultados.csv"
	results_frame.to_csv(output_path, index=False)
	return output_path


def save_performance_plot(results_frame: pd.DataFrame) -> Path:
	plot_path = OUTPUT_DIRECTORY / "knn_desempenho.png"

	plt.style.use("seaborn-v0_8-whitegrid")
	figure, axis = plt.subplots(figsize=(10, 6))

	for column_name in ["Euclidiana", "Manhattan", "Chebyshev", "Minkowski"]:
		axis.plot(results_frame["K"], results_frame[column_name], marker="o", linewidth=2, label=column_name)

	axis.set_title("Acurácia por valor de K e métrica de distância")
	axis.set_xlabel("Valor de K")
	axis.set_ylabel("Acurácia no conjunto de teste")
	axis.set_xticks(range(1, 16))
	axis.set_ylim(0.2, 0.8)
	axis.legend()
	axis.grid(True, alpha=0.3)

	figure.tight_layout()
	figure.savefig(plot_path, dpi=200)
	plt.close(figure)

	return plot_path


def main() -> None:
	dataframe = load_dataset()
	print(f"Dataset carregado com {dataframe.shape[0]} linhas e {dataframe.shape[1]} colunas.")
	print(f"Valores ausentes: {int(dataframe.isna().sum().sum())}")
	print("Distribuição da classe-alvo (Ano):")
	print(dataframe[TARGET_COLUMN].value_counts().sort_index().to_string())

	feature_frame = dataframe.drop(columns=[TARGET_COLUMN])
	target_series = dataframe[TARGET_COLUMN]
	results_frame = evaluate_models(feature_frame, target_series)

	results_path = save_results_table(results_frame)
	plot_path = save_performance_plot(results_frame)

	print("\nTabela de resultados:")
	print(results_frame.to_string(index=False))
	print(f"\nResultados salvos em: {results_path}")
	print(f"Gráfico salvo em: {plot_path}")

	best_position = results_frame.set_index("K")[["Euclidiana", "Manhattan", "Chebyshev", "Minkowski"]].stack().idxmax()
	best_k, best_metric = best_position
	best_accuracy = results_frame.loc[results_frame["K"] == best_k, best_metric].iloc[0]
	print(f"Melhor combinação: K={best_k}, métrica={best_metric}, acurácia={best_accuracy:.4f}")


if __name__ == "__main__":
	main()