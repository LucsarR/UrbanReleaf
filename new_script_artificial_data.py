import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Create artificial data
def create_artificial_data(num_samples=100, image_size=(10, 10)):
    X = []
    y = []
    for _ in range(num_samples):
        # Generate random grayscale image data
        image = np.random.rand(*image_size).flatten()
        X.append(image)
        # Generate random target values
        target = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
        y.append(target)
    return np.array(X), np.array(y)

# Main script
if __name__ == "__main__":
    # Generate artificial data
    X, y = create_artificial_data(num_samples=500, image_size=(10, 10))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize SGDRegressor for linear regression
    model = SGDRegressor(max_iter=1, tol=None, learning_rate='constant', eta0=0.01, random_state=42)

    # Train the model manually and track loss per iteration
    losses = []
    for epoch in range(100):
        model.partial_fit(X_train, y_train[:, 0])  # Train for the first target component (a)
        y_pred_train = model.predict(X_train)
        loss = mean_squared_error(y_train[:, 0], y_pred_train)
        losses.append(loss)

    # Plot the loss function
    plt.plot(range(1, 101), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Function Over Iterations')
    plt.show()

    # Evaluate the model on the test set
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test[:, 0], y_pred_test)
    print(f"Test Mean Squared Error for first target component: {mse_test}")
