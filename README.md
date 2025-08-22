Churn Prediction App 📱
This repository contains a full-stack machine learning application designed to predict customer churn for a telecommunications company. The app is built using Python, with a focus on a robust data pipeline, a powerful XGBoost model, and clear model interpretability.

🚀 Key Features
Robust Data Preprocessing: A dedicated dataloader.py handles data ingestion and cleaning, ensuring a consistent and reliable input for the model. The pre-trained preprocessor.pkl saves the preprocessing steps, so the exact transformations can be applied consistently to new data.

Optimized XGBoost Model: The core of the app is a powerful XGBoost Classifier. The model's hyperparameters are automatically tuned using RandomizedSearchCV to achieve optimal predictive performance.

Model Interpretability: Utilizes SHAP (SHapley Additive exPlanations) to provide clear, human-readable insights into what drives the model's predictions. The app can generate both global and local explanations, helping to understand which features contribute most to churn predictions.

Modular Codebase: The project is organized into separate files for data handling (dataloader.py), general utilities (utils.py), model training and evaluation (train.py), and model explanation (explain.py), making the code easy to read, maintain, and extend.

🛠️ How to Run the App
Clone the Repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo
Install Dependencies:
All required libraries are listed in requirements.txt.

pip install -r requirements.txt
Run the Main Script:
The main app.py script orchestrates the entire process, from data loading to model training and analysis.

python app.py
📊 Project Structure
app.py: The main entry point for the application. It calls functions from other modules to execute the end-to-end workflow.

dataloader.py: Handles all data-related tasks, including loading, cleaning, and splitting.

utils.py: Contains a collection of helper functions used throughout the project.

train.py: Contains the core logic for building the model pipeline, training, and evaluating its performance.

explain.py: Generates the SHAP-based model explanations to visualize feature importance.

requirements.txt: Lists all Python libraries required to run the project.

preprocessor.pkl: The saved ColumnTransformer object, which is used to preprocess new data consistently.

churn_model.pkl: The saved, trained model pipeline.