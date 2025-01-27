import joblib

# Function to load the model
def load_model(model_path):
    model = joblib.load(model_path)  # or pickle.load() for .pkl files
    return model
