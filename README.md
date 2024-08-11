# Crop Recommendation System

This project is a Crop Recommendation System that uses machine learning to recommend the best crop to plant based on various soil and weather parameters.

## Features

- **Data Loading and Preprocessing**: Load and preprocess the crop recommendation dataset.
- **Model Training**: Train a machine learning model to predict the best crop.
- **Web Interface**: A simple web interface to input soil and weather parameters and get crop recommendations.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ashutoshpatraa/Crop-recommendation-.git
    cd Crop-recommendation-
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Web Application

1. Ensure you have the `templates` directory with `index.html` inside it:
    ```
    /path/to/your/project/
    ├── app.py
    └── templates/
        └── index.html
    ```

2. Start the Flask web application:
    ```sh
    python app.py
    ```

3. Open your web browser and go to `http://127.0.0.1:5000/` to access the web interface.

### Running the Crop Recommendation Script

1. Run the [`crop.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fashu%2FDocuments%2FGitHub%2FCrop-recommendation-%2Fcrop.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\ashu\Documents\GitHub\Crop-recommendation-\crop.py") script to load data, preprocess it, and train the model:
    ```sh
    python crop.py
    ```

## File Structure

- [`app.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fashu%2FDocuments%2FGitHub%2FCrop-recommendation-%2Fapp.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\ashu\Documents\GitHub\Crop-recommendation-\app.py"): The main Flask application file.
- [`crop.py`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fashu%2FDocuments%2FGitHub%2FCrop-recommendation-%2Fcrop.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\ashu\Documents\GitHub\Crop-recommendation-\crop.py"): The script for data loading, preprocessing, and model training.
- [`templates/index.html`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fashu%2FDocuments%2FGitHub%2FCrop-recommendation-%2Ftemplates%2Findex.html%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\ashu\Documents\GitHub\Crop-recommendation-\templates\index.html"): The HTML template for the web interface.
- `requirements.txt`: The list of required Python packages.

## Dependencies

- Flask
- Pandas
- Numpy
- Matplotlib
- TensorFlow
- Scikit-learn

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License.