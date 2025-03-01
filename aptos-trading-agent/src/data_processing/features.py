import pandas as pd
import numpy as np
from src.utils.logger import logger

def extract_features(transactions: list, pool_data: dict) -> pd.DataFrame:
    """
    Extract relevant features from transaction data and pool information.
    Transforms blockchain data into ML features.
    """
    if not transactions:
        logger.warning("No transactions to extract features from")
        return pd.DataFrame()

    features = []
    tx_timestamps = [tx.get('timestamp', 0) for tx in transactions if 'timestamp' in tx]
    if not tx_timestamps:
        return pd.DataFrame()

    time_features = {
        'tx_count': len(transactions),
        'unique_addresses': len(set(tx.get('sender', '') for tx in transactions)),
        'avg_gas_used': np.mean([int(tx.get('gas_used', 0)) for tx in transactions]),
        'median_gas_used': np.median([int(tx.get('gas_used', 0)) for tx in transactions]),
        'tx_volume': sum([int(tx.get('gas_price', 0)) * int(tx.get('gas_used', 0)) for tx in transactions]),
    }

    # Extract function call counts
    function_calls = {}
    for tx in transactions:
        if 'payload' in tx and 'function' in tx['payload']:
            func = tx['payload']['function']
            function_calls[func] = function_calls.get(func, 0) + 1

    top_functions = sorted(function_calls.items(), key=lambda x: x[1], reverse=True)[:5]
    for func, count in top_functions:
        func_name = func.split('::')[-1]
        time_features[f'func_{func_name}'] = count

    # Include pool data if available
    if pool_data and 'data' in pool_data and 'coin' in pool_data['data']:
        time_features['pool_value'] = int(pool_data['data']['coin']['value'])

    features.append(time_features)
    return pd.DataFrame(features)
