import os
import time
import json
from src.trading.trader import AptosTrader
from src.train_model import train_model
from src.config.settings import NETWORK
from src.utils.logger import logger

def main():
    try:
        model_dir = "models"
        data_path = "historical_data.csv"  # Ensure this file exists with training data

        if not os.path.exists(os.path.join(model_dir, 'trading_model.joblib')):
            logger.info("No existing model found, training new model")
            model_path, scaler_path = train_model(data_path, model_dir)
        else:
            model_path = os.path.join(model_dir, 'trading_model.joblib')
            scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
            logger.info("Using existing model")

        trader = AptosTrader(network=NETWORK)
        trader.load_model(model_path, scaler_path)

        pool_address = "0x05a97986a9d031c4567e15b797be516910cfcb4156312482efc6a19c0a30c948"  # Example pool
        logger.info("Starting trading loop")
        while True:
            try:
                trader.run_trading_cycle(pool_address=pool_address)
                with open("performance_metrics.json", "w") as f:
                    json.dump(trader.calculate_performance(), f, indent=2)
                time.sleep(300)  # 5 minutes delay between cycles
            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {str(e)}")
                time.sleep(60)
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()
