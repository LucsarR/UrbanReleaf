import os
from dotenv import load_dotenv
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox
from rasterio.transform import from_bounds
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

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

    swir_evalscript = """
    // Script para SWIR usando Sentinel-2 (Band 11 e Band 12)
    function setup() {
        return {
            input: ["B11", "B12"], // SWIR 1 e SWIR 2
            output: { bands: 2 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B11, sample.B12];
    }
    """

    # Defining the area of interest (São Paulo)
    region_name = 'SaoPaulo'
    bbox = [-46.693419, -23.568704, -46.623049, -23.511217]
    bbox = BBox(bbox, crs=CRS.WGS84)

    # Defining the time interval
    time_interval = ('2021-01-01', '2021-01-31')

    # Create a directory to store the data
    os.makedirs('data', exist_ok=True)

    # Update the evalscripts to use the correct bands and resolutions
    ndvi_resolution = '10m'    # Bands B04 and B08 (Sentinel-2) have a resolution of 10m
    swir_resolution = '20m'    # Bands B11 and B12 (Sentinel-2) have a resolution of 20m
    lst_resolution = '100m'    # Band B10 (Landsat 8) has a resolution of 100m

    date_str = time_interval[1].replace('-', '')  # Remove hyphens from the date and using final date

    # Define the filenames
    ndvi_filename = f"NDVI_{ndvi_resolution}_{date_str}_{region_name}.tif"
    lst_filename = f"LST_{lst_resolution}_{date_str}_{region_name}.tif"
    swir_filename = f"SWIR_{swir_resolution}_{date_str}_{region_name}.tif"

    ndvi_tiff_path = os.path.join('data', ndvi_filename)
    lst_tiff_path = os.path.join('data', lst_filename)
    swir_tiff_path = os.path.join('data', swir_filename)

    ndvi_data_collection = DataCollection.SENTINEL2_L1C
    swir_data_collection = DataCollection.SENTINEL2_L1C
    lst_data_collection = DataCollection.LANDSAT_OT_L1 

    if not os.path.exists(ndvi_tiff_path):
        ndvi_tiff_path = download_tiff_from_sentinelhub(
            bbox, time_interval, config, ndvi_filename, ndvi_evalscript, ndvi_data_collection)
    if not os.path.exists(lst_tiff_path):
        lst_tiff_path = download_tiff_from_sentinelhub(
            bbox, time_interval, config, lst_filename, lst_evalscript, lst_data_collection)
    if not os.path.exists(swir_tiff_path):
        swir_tiff_path = download_tiff_from_sentinelhub(
            bbox, time_interval, config, swir_filename, swir_evalscript, swir_data_collection)

    # Read NDVI and LST data
    with rasterio.open(ndvi_tiff_path) as src:
        ndvi = src.read(1).flatten()
    
    with rasterio.open(lst_tiff_path) as src:
        lst = src.read(1).flatten()
    
    # Prepare the dataset
    # Reshape NDVI for sklearn and handle possible NaN values
    valid_indices = ~np.isnan(ndvi) & ~np.isnan(lst)
    X = ndvi[valid_indices].reshape(-1, 1)
    y = lst[valid_indices]
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle outliers by removing data points where NDVI is beyond 3 standard deviations
    mean_ndvi = np.mean(X_scaled)
    std_ndvi = np.std(X_scaled)
    outlier_indices = np.abs(X_scaled - mean_ndvi) > 3 * std_ndvi
    outlier_indices = outlier_indices.flatten()  # Flatten to 1D
    X_filtered = X_scaled[~outlier_indices].reshape(-1, 1)
    y_filtered = y[~outlier_indices]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42
    )
    
    # Define the models to be trained
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),  # L2 regularization
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        )
    }
    
    # Train and evaluate each model with cross-validation
    print("Model Evaluation:")
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        model.fit(X_train, y_train)
        mse, r2, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"\n{name}:")
        print(f"  Cross-Validated R² Scores: {cv_scores}")
        print(f"  Average Cross-Validated R²: {cv_scores.mean():.2f}")
        print(f"  Mean Squared Error (MSE): {mse:.2f}")
        print(f"  R² Score on Test Set: {r2:.2f}")
    
    # Save trained models
    models_dir = os.path.join('results', 'models')
    os.makedirs(models_dir, exist_ok=True)  # Create models directory if it doesn't exist

    for name, model in models.items():
        model_filename = f"{name.replace(' ', '_').lower()}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        dump(model, model_path)
        print(f"Saved {name} model to {model_path}")

if __name__ == '__main__':
    main()