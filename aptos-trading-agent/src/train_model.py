import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger

def train_model(data_path: str, output_dir: str):
    logger.info(f"Training model with data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        X = df.drop('signal', axis=1)
        y = df['signal']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_scaled, y)
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'trading_model.joblib')
        scaler_path = os.path.join(output_dir, 'feature_scaler.joblib')
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        return model_path, scaler_path
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python train_model.py <data_path> <output_dir>")
        sys.exit(1)
    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    train_model(data_path, output_dir)
