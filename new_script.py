import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Path to the folder containing .tiff files
folder_path = "path_to_tiff_folder"

# Dummy function to simulate loading target vectors for each image
def load_targets(file_names):
    # Replace this with your actual method to get target vectors for each file
    targets = []
    for file_name in file_names:
        # Example: Generate random target vectors (replace with actual targets)
        targets.append([np.random.uniform(0, 1), np.random.uniform(0, 1)])
    return np.array(targets)

# Load image data and corresponding target vectors
def load_data(folder_path):
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]
    images = []

    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        # Read the image and convert it to a flattened array
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image.flatten())

    images = np.array(images)
    targets = load_targets(file_names)
    return images, targets

# Main script
if __name__ == "__main__":
    # Load data
    X, y = load_data(folder_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Example: Print actual vs predicted values for the first few samples
    for i in range(min(5, len(y_test))):
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
