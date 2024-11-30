import os
import rasterio
import matplotlib.pyplot as plt

def visualize_tiff(file_path, save_dir):
    """
    Visualize a .tiff file and save the plot as a .png in the specified directory.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis')
        plt.colorbar()
        plt.title(os.path.basename(file_path))
        plt.axis('off')
        # Save the figure
        filename = os.path.splitext(os.path.basename(file_path))[0] + '.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()

def main():
    data_dir = 'data'
    # Define the results directory
    results_dir = os.path.join('results', 'tiff_viewer')
    os.makedirs(results_dir, exist_ok=True)  # Create directories if they don't exist

    # List all .tif and .tiff files in the data directory
    tiff_files = [f for f in os.listdir(data_dir) if f.endswith(('.tif', '.tiff'))]

    for tiff_file in tiff_files:
        file_path = os.path.join(data_dir, tiff_file)
        print(f"Processing {tiff_file}...")
        visualize_tiff(file_path, results_dir)

if __name__ == '__main__':
    main()
