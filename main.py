import os
from dotenv import load_dotenv
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox

# Carregar variáveis de ambiente
load_dotenv()


def load_tiff_as_array(file_path):
    """
    Carrega um arquivo .tiff e retorna como um array numpy.
    """
    with rasterio.open(file_path) as src:
        return src.read(1)  # Lê apenas a primeira banda como array 2D


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Treina e avalia um modelo.
    """
    # Treinamento do modelo
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o desempenho do modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_pred


def download_tiff_from_sentinelhub(bbox, time_interval, config):
    """
    Baixa um arquivo .tiff do Sentinel Hub.
    """
    request = SentinelHubRequest(
        data_folder='./data',
        evalscript="""
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: {
                    bands: 3,
                    sampleType: "UINT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B02, sample.B03, sample.B04];
        }
        """,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=BBox(bbox, crs=CRS.WGS84),
        size=(512, 512),
        config=config
    )

    response = request.get_data(save_data=True)
    return response[0]['default.tif']


def main():

    # Configura as credenciais do Sentinel Hub
    config = SHConfig()
    config.instance_id = os.getenv('INSTANCE_ID')
    config.sh_client_id = os.getenv('CLIENT_ID')
    config.sh_client_secret = os.getenv('CLIENT_SECRET')
    if not config.instance_id or not config.sh_client_id or not config.sh_client_secret:
        raise ValueError("Configuração incompleta. Verifique Instance ID, Client ID e Client Secret.")

    # Definindo a área de interesse (São Paulo)
    bbox = BBox(bbox=[-46.693419, -23.568704, -46.623049, -23.511217], crs=CRS.WGS84)

    # Definindo o intervalo de tempo
    time_interval = ('2021-01-01', '2021-01-31')

    # Baixar imagens do Sentinel Hub
    ndvi_tiff_path = download_tiff_from_sentinelhub(bbox, time_interval, config)
    lst_tiff_path = download_tiff_from_sentinelhub(bbox, time_interval, config)
    swir_tiff_path = download_tiff_from_sentinelhub(bbox, time_interval, config)

    # Carregar os dados como arrays
    ndvi_matrix = load_tiff_as_array(ndvi_tiff_path)
    lst_matrix = load_tiff_as_array(lst_tiff_path)
    swir_matrix = load_tiff_as_array(swir_tiff_path)

    # Vetorizar os dados
    ndvi_vector = ndvi_matrix.flatten()
    lst_vector = lst_matrix.flatten()
    swir_vector = swir_matrix.flatten()

    # Combinar NDVI e SWIR como features
    X = np.vstack([ndvi_vector, swir_vector]).T
    y = lst_vector

    # Remover dados inválidos
    #valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1) & (y > 0) & (X.min(axis=1) >= 0)
    #X = X[valid_mask]
    #y = y[valid_mask]

    # Divisão do conjunto de dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelos a serem testados
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),  # Regularização L2 para evitar overfitting
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Avaliação de cada modelo
    print("Avaliação dos Modelos:")
    for name, model in models.items():
        mse, r2, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"\n{name}:")
        print(f"  Mean Squared Error (MSE): {mse:.2f}")
        print(f"  R² Score: {r2:.2f}")

if __name__ == '__main__':
    main()