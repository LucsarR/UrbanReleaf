import os
from dotenv import load_dotenv
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor  # Corrected module name
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
from joblib import dump
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train the model and evaluate the performance based on MSE and R².
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_pred

def download_tiff_from_sentinelhub(bbox, time_interval, config, filename, evalscript, data_collection):
    """
    Download a .tiff file from Sentinel Hub using the specified evalscript and data collection.
    """
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=time_interval
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)  # Request raw bytes
        ],
        bbox=bbox,
        size=(512, 512),
        config=config
    )
    response = request.get_data(decode_data=False)  # Get raw bytes
    file_path = os.path.join('data', filename)
    with open(file_path, 'wb') as f:
        f.write(response[0].content)  # Write raw bytes to file
    return file_path

def main():
    # Sentinel Hub configuration
    config = SHConfig()
    config.instance_id = os.getenv('INSTANCE_ID')
    config.sh_client_id = os.getenv('CLIENT_ID')
    config.sh_client_secret = os.getenv('CLIENT_SECRET')
    if not config.instance_id or not config.sh_client_id or not config.sh_client_secret:
        raise ValueError("Configuração incompleta. Verifique Instance ID, Client ID e Client Secret.")

    # NDVI, LST, and SWIR evalscripts
    ndvi_evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B08"],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
        return [ndvi];
    }
    """

    lst_evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B10"],
                units: ["BRIGHTNESS_TEMPERATURE"]
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        // sample.B10 is in Kelvin
        var temperature_celsius = sample.B10 - 273.15; // Convert to Celsius
        return [temperature_celsius];
    }
    """

    # swir_evalscript = """
    # // Script para SWIR usando Sentinel-2 (Band 11 e Band 12)
    # function setup() {
    #     return {
    #         input: ["B11", "B12"], // SWIR 1 e SWIR 2
    #         output: { bands: 2 }
    #     };
    # }
    #
    # function evaluatePixel(sample) {
    #     return [sample.B11, sample.B12];
    # }
    # """

    # Defining the area of interest (São Paulo)
    region_name = 'SaoPaulo'
    bbox = [-46.76, -23.65, -46.55, -23.43]
    bbox = BBox(bbox, crs=CRS.WGS84)

    # Defining the time interval
    time_interval = ('2020-01-01', '2022-12-31')

    # Create a directory to store the data
    os.makedirs('data', exist_ok=True)

    # Update the evalscripts to use the correct bands and resolutions
    ndvi_resolution = '10m'    # Bands B04 and B08 (Sentinel-2) have a resolution of 10m
    lst_resolution = '100m'    # Band B10 (Landsat 8) has a resolution of 100m

    date_str = time_interval[1].replace('-', '')  # Remove hyphens from the date and using final date

    # Define the filenames
    ndvi_filename = f"NDVI_{ndvi_resolution}_{date_str}_{region_name}.tif"
    lst_filename = f"LST_{lst_resolution}_{date_str}_{region_name}.tif"

    ndvi_tiff_path = os.path.join('data', ndvi_filename)
    lst_tiff_path = os.path.join('data', lst_filename)

    ndvi_data_collection = DataCollection.SENTINEL2_L1C
    lst_data_collection = DataCollection.LANDSAT_OT_L1 

    # swir_resolution = '20m'    # Bands B11 and B12 (Sentinel-2) have a resolution of 20m
    # swir_filename = f"SWIR_{swir_resolution}_{date_str}_{region_name}.tif"

    # swir_tiff_path = os.path.join('data', swir_filename)

    if not os.path.exists(ndvi_tiff_path):
        ndvi_tiff_path = download_tiff_from_sentinelhub(
            bbox, time_interval, config, ndvi_filename, ndvi_evalscript, ndvi_data_collection)
    if not os.path.exists(lst_tiff_path):
        lst_tiff_path = download_tiff_from_sentinelhub(
            bbox, time_interval, config, lst_filename, lst_evalscript, lst_data_collection)

    # swir_data_collection = DataCollection.SENTINEL2_L1C

    # if not os.path.exists(swir_tiff_path):
    #     swir_tiff_path = download_tiff_from_sentinelhub(
    #         bbox, time_interval, config, swir_filename, swir_evalscript, swir_data_collection)

    # Define the results directory for graphs
    graphs_dir = os.path.join('results', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)  # Create graphs directory if it doesn't exist

    # Load and preprocess the data
    def load_data(tiff_paths):
        X = []
        ndvi_path = tiff_paths[0]
        lst_path = tiff_paths[1]
        
        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(1).flatten()
            X.append(ndvi)
        
        with rasterio.open(lst_path) as src:
            lst = src.read(1).flatten()
        
        # Verify that NDVI and LST images have the same dimensions
        with rasterio.open(ndvi_path) as src_ndvi, rasterio.open(lst_path) as src_lst:
            assert src_ndvi.width == src_lst.width and src_ndvi.height == src_lst.height, "NDVI and LST images have different dimensions."
        
        # Remove any rows with NaN or infinite values
        valid_indices = ~np.isnan(X[0]) & ~np.isinf(X[0]) & ~np.isnan(lst) & ~np.isinf(lst)
        X = np.array(X).T[valid_indices]
        y = lst[valid_indices]

        return X, y

    # Update tiff_paths
    tiff_paths = [ndvi_tiff_path, lst_tiff_path]
    X, y = load_data(tiff_paths)

    # Check data shapes
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # Plot a sample pair to verify alignment
    plt.scatter(X[:100, 0], y[:100], alpha=0.5)
    plt.xlabel('NDVI')
    plt.ylabel('LST (°C)')
    plt.title('Sample NDVI vs. LST')
    plt.savefig(os.path.join(graphs_dir, 'sample_ndvi_vs_lst.png'))
    plt.close()

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optionally scale the target variable
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model as a baseline
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    # Inverse transform the baseline predictions to the original scale
    y_pred_baseline = scaler_y.inverse_transform(y_pred_baseline.reshape(-1, 1)).ravel()
    # Inverse transform the scaled test targets for evaluation
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mse_baseline = mean_squared_error(y_test_inv, y_pred_baseline)
    r2_baseline = r2_score(y_test_inv, y_pred_baseline)
    print(f"Baseline Test Mean Squared Error: {mse_baseline}")
    print(f"Baseline Test R² Score: {r2_baseline}")

    # Initialize SGDRegressor with adjusted hyperparameters
    model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01, random_state=42)

    # Train the model and track loss over iterations
    losses = []
    for epoch in range(100):
        model.partial_fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        loss = mean_squared_error(y_train, y_pred_train)
        losses.append(loss)

    y_pred_test = model.predict(X_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    mse_test = mean_squared_error(y_test_inv, y_pred_test)
    r2_test = r2_score(y_test_inv, y_pred_test)
    print(f"SGDRegressor Test Mean Squared Error: {mse_test}")
    print(f"SGDRegressor Test R² Score: {r2_test}")

    # Plot the loss function over iterations
    plt.plot(range(1, 101), losses, label=f'SGDRegressor Loss\nR²: {r2_test:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('SGDRegressor Loss Over Iterations')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, 'sgd_loss_over_iterations.png'))
    plt.close()

    # Plot Predicted vs Actual LST for SGDRegressor
    plt.scatter(y_test_inv, y_pred_test, alpha=0.5)
    plt.xlabel('Actual LST (°C)')
    plt.ylabel('Predicted LST (°C)')
    plt.title('SGDRegressor: Actual vs. Predicted LST')
    plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
    plt.savefig(os.path.join(graphs_dir, 'sgd_actual_vs_predicted.png'))
    plt.close()

    # Compare with Baseline
    plt.scatter(y_test_inv, y_pred_baseline, alpha=0.5, label='LinearRegression')
    plt.scatter(y_test_inv, y_pred_test, alpha=0.5, label='SGDRegressor')
    plt.xlabel('Actual LST (°C)')
    plt.ylabel('Predicted LST (°C)')
    plt.title('Actual vs. Predicted LST Comparison')
    plt.legend()
    plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
    plt.savefig(os.path.join(graphs_dir, 'actual_vs_predicted_comparison.png'))
    plt.close()

    # Export trained models
    models_dir = os.path.join('results', 'models')
    os.makedirs(models_dir, exist_ok=True)
    dump(baseline_model, os.path.join(models_dir, 'linear_regression_model.joblib'))
    dump(model, os.path.join(models_dir, 'sgd_regressor_model.joblib'))
    
    # Export the scalers
    scalers_dir = os.path.join('results', 'scalers')
    os.makedirs(scalers_dir, exist_ok=True)
    dump(scaler, os.path.join(scalers_dir, 'scaler.joblib'))
    dump(scaler_y, os.path.join(scalers_dir, 'scaler_y.joblib'))

if __name__ == '__main__':
    main()