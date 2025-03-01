import os
import time
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

from aptos_sdk.account import Account
from aptos_sdk.client import RestClient
from aptos_sdk.transactions import EntryFunction, TransactionArgument, TransactionPayload
from aptos_sdk.bcs import Serializer

from src.config.settings import PRIVATE_KEY, API_ENDPOINTS, TOKEN_CONFIGS
from src.data_processing.transactions import fetch_transactions, fetch_pool_data
from src.data_processing.features import extract_features
from src.utils.logger import logger

class AptosTrader:
    def __init__(self, network: str = "testnet"):
        self.network = network
        self.api_url = API_ENDPOINTS.get(network)
        self.client = RestClient(self.api_url)

        if not PRIVATE_KEY:
            raise ValueError("APTOS_PRIVATE_KEY environment variable must be set")
        try:
            self.account = Account.load_key(PRIVATE_KEY)
            logger.info(f"Account initialized: {self.account.address()}")
        except Exception as e:
            logger.error(f"Failed to initialize account: {str(e)}")
            raise

        self.model = None
        self.scaler = None
        self.trades = []
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "profit_loss": 0.0,
        }

    def load_model(self, model_path: str, scaler_path: str) -> None:
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        if features_df.empty:
            logger.warning("Empty features dataframe, cannot preprocess")
            return np.array([])
        try:
            # Ensure all expected columns are present
            for col in self.model.feature_names_in_:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.model.feature_names_in_]
            scaled_features = self.scaler.transform(features_df)
            return scaled_features
        except Exception as e:
            logger.error(f"Feature preprocessing failed: {str(e)}")
            return np.array([])

    def generate_signal(self, features: np.ndarray) -> Tuple[str, float]:
        if len(features) == 0:
            return "NO_SIGNAL", 0.0
        try:
            probabilities = self.model.predict_proba(features)[0]
            class_labels = self.model.classes_
            max_prob_idx = np.argmax(probabilities)
            max_prob = probabilities[max_prob_idx]
            predicted_class = class_labels[max_prob_idx]
            if max_prob < 0.6:
                return "NO_SIGNAL", max_prob
            return predicted_class, max_prob
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            return "NO_SIGNAL", 0.0

    def execute_trade(self, signal: str, confidence: float, amount: float) -> bool:
        if signal == "NO_SIGNAL":
            logger.info("No trade signal generated")
            return False
        if signal == "HOLD":
            logger.info("Signal is HOLD, no action needed")
            return True
        try:
            logger.info(f"Executing {signal} trade with confidence {confidence:.2f} for {amount} tokens")
            trade_info = {
                "timestamp": datetime.now().isoformat(),
                "signal": signal,
                "confidence": confidence,
                "amount": amount,
                "status": "PENDING"
            }
            if signal == "BUY":
                payload = self._create_swap_payload("APT", "OTHER_TOKEN", amount)
            else:
                payload = self._create_swap_payload("OTHER_TOKEN", "APT", amount)
            tx_hash = self._submit_transaction(payload)
            if tx_hash:
                trade_info["status"] = "COMPLETED"
                trade_info["tx_hash"] = tx_hash
                self.performance_metrics["successful_trades"] += 1
                logger.info(f"Trade executed successfully: {tx_hash}")
                success = True
            else:
                trade_info["status"] = "FAILED"
                self.performance_metrics["failed_trades"] += 1
                logger.error("Trade execution failed")
                success = False
            self.trades.append(trade_info)
            self.performance_metrics["total_trades"] += 1
            return success
        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            self.performance_metrics["failed_trades"] += 1
            return False

    def _create_swap_payload(self, from_token: str, to_token: str, amount: float) -> TransactionPayload:
        token_x_type = f"{TOKEN_CONFIGS[from_token]['module_address']}::{TOKEN_CONFIGS[from_token]['module_name']}::{TOKEN_CONFIGS[from_token]['struct_name']}"
        token_y_type = f"{TOKEN_CONFIGS[to_token]['module_address']}::{TOKEN_CONFIGS[to_token]['module_name']}::{TOKEN_CONFIGS[to_token]['struct_name']}"
        amount_as_int = int(amount * 10**8)
        return TransactionPayload(
            EntryFunction.natural(
                "dex_module_address::dex_module::swap",  # Replace with actual module details
                [],
                [
                    TransactionArgument(amount_as_int, Serializer.u64),
                    TransactionArgument(0, Serializer.u64)
                ]
            )
        )

    def _submit_transaction(self, payload: TransactionPayload) -> Optional[str]:
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                tx_hash = self.client.submit_transaction(self.account, payload)
                self.client.wait_for_transaction(tx_hash)
                return tx_hash
            except Exception as e:
                logger.warning(f"Transaction attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        return None

    def calculate_performance(self) -> Dict:
        metrics = self.performance_metrics.copy()
        if metrics["total_trades"] > 0:
            metrics["success_rate"] = metrics["successful_trades"] / metrics["total_trades"]
        else:
            metrics["success_rate"] = 0.0
        return metrics

    def run_trading_cycle(self, pool_address: str = None, trade_amount: float = 1.0) -> None:
        if not self.model or not self.scaler:
            logger.error("Model not loaded, cannot run trading cycle")
            return
        transactions = fetch_transactions(self.api_url, self.account.address())
        pool_data = {}
        if pool_address:
            pool_data = fetch_pool_data(self.api_url, pool_address)
        features_df = extract_features(transactions, pool_data)
        if features_df.empty:
            logger.warning("No features extracted, skipping trading cycle")
            return
        scaled_features = self.preprocess_features(features_df)
        if len(scaled_features) == 0:
            logger.warning("Feature preprocessing failed, skipping trading cycle")
            return
        signal, confidence = self.generate_signal(scaled_features)
        logger.info(f"Generated signal: {signal} with confidence {confidence:.2f}")
        if signal in ["BUY", "SELL"]:
            self.execute_trade(signal, confidence, trade_amount)
        metrics = self.calculate_performance()
        logger.info(f"Current performance: {json.dumps(metrics)}")
