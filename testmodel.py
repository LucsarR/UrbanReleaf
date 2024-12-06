import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

def load_scalers(scalers_dir):
    """
    Load the saved StandardScaler objects.
    """
    scaler_path = os.path.join(scalers_dir, 'scaler.joblib')
    scaler_y_path = os.path.join(scalers_dir, 'scaler_y.joblib')
    scaler = load(scaler_path)
    scaler_y = load(scaler_y_path)
    return scaler, scaler_y

def preprocess_ndvi(ndvi_path, scaler):
    """
    Load and preprocess the NDVI .tiff file.
    """
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1).flatten()
    
    # Handle NaN or infinite values if any
    valid_indices = ~np.isnan(ndvi) & ~np.isinf(ndvi)
    ndvi = ndvi[valid_indices]
    
    # Reshape for scaler
    ndvi = ndvi.reshape(-1, 1)
    
    # Scale the NDVI values
    ndvi_scaled = scaler.transform(ndvi)
    
    return ndvi_scaled, valid_indices

def predict_lst(model, ndvi_scaled):
    """
    Use the trained model to predict LST from scaled NDVI data.
    """
    lst_scaled_pred = model.predict(ndvi_scaled)
    return lst_scaled_pred

def inverse_scale_predictions(lst_scaled_pred, scaler_y):
    """
    Inverse transform the scaled LST predictions to original scale.
    """
    lst_pred = scaler_y.inverse_transform(lst_scaled_pred.reshape(-1, 1)).ravel()
    return lst_pred

def save_predicted_lst(lst_pred, save_path, original_shape, transform):
    """
    Save the predicted LST as a .tiff file.
    """
    lst_pred_reshaped = lst_pred.reshape(original_shape)
    
    # Define metadata for the new .tiff file
    new_meta = {
        'driver': 'GTiff',
        'height': original_shape[0],
        'width': original_shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform  # Use the original transform from the NDVI image
    }
    
    with rasterio.open(save_path, 'w', **new_meta) as dst:
        dst.write(lst_pred_reshaped.astype('float32'), 1)

def visualize_predicted_lst(lst_pred, save_dir, filename):
    """
    Visualize the predicted LST and save the plot as a .png file.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(lst_pred, cmap='coolwarm', vmin=0, vmax=35)
    plt.colorbar(label='LST (°C)')
    plt.title('Predicted Land Surface Temperature (LST)')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{filename}_predicted_lst.png"))
    plt.close()

def main():
    # Paths
    models_dir = os.path.join('results', 'models')
    scalers_dir = os.path.join('results', 'scalers')  # Directory to store scalers
    test_data_dir = 'testdata'  # Updated from 'test_data' to 'testdata'
    save_dir = os.path.join('results', 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    # Specify the NDVI file for prediction
    # Search for the NDVI file in the test data directory
    ndvi_filename = None
    for file in os.listdir(test_data_dir):
        if file.endswith('.tif'):
            ndvi_filename = file
            break
    if ndvi_filename is None:
        raise FileNotFoundError("No NDVI .tiff file found in the test data directory.")
    ndvi_path = os.path.join(test_data_dir, ndvi_filename)
    
    # Load the trained models
    linear_model_path = os.path.join(models_dir, 'linear_regression_model.joblib')
    sgd_model_path = os.path.join(models_dir, 'sgd_regressor_model.joblib')
    linear_model = load(linear_model_path)
    sgd_model = load(sgd_model_path)
    
    # Load the scalers
    scaler, scaler_y = load_scalers(scalers_dir)
    
    # Preprocess the NDVI data
    ndvi_scaled, valid_indices = preprocess_ndvi(ndvi_path, scaler)
    
    # Retrieve the original transform from the NDVI image
    with rasterio.open(ndvi_path) as src:
        original_transform = src.transform
    
    # Predict LST using Linear Regression
    lst_pred_linear_scaled = linear_model.predict(ndvi_scaled)
    lst_pred_linear = inverse_scale_predictions(lst_pred_linear_scaled, scaler_y)
    
    # Predict LST using SGD Regressor
    lst_pred_sgd_scaled = sgd_model.predict(ndvi_scaled)
    lst_pred_sgd = inverse_scale_predictions(lst_pred_sgd_scaled, scaler_y)
    
    # Save the predicted LST as .tiff files
    original_shape = (512, 512)  # Replace with the original shape of your NDVI images
    save_path_linear = os.path.join(save_dir, f"{os.path.splitext(ndvi_filename)[0]}_predicted_lst_linear.tif")
    save_path_sgd = os.path.join(save_dir, f"{os.path.splitext(ndvi_filename)[0]}_predicted_lst_sgd.tif")
    save_predicted_lst(lst_pred_linear, save_path_linear, original_shape, original_transform)
    save_predicted_lst(lst_pred_sgd, save_path_sgd, original_shape, original_transform)
    
    # Visualize the predicted LST
    visualize_predicted_lst(lst_pred_linear.reshape(original_shape), save_dir, f"{os.path.splitext(ndvi_filename)[0]}_linear")
    visualize_predicted_lst(lst_pred_sgd.reshape(original_shape), save_dir, f"{os.path.splitext(ndvi_filename)[0]}_sgd")

if __name__ == '__main__':
    main()
