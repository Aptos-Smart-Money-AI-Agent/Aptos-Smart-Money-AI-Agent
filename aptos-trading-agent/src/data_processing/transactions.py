import time
import requests
from typing import List, Dict
from src.config.settings import API_ENDPOINTS
from src.utils.logger import logger

def fetch_transactions(api_url: str, address: str, limit: int = 100) -> List[Dict]:
    """Fetch recent transactions with retry logic."""
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            url = f"{api_url}/accounts/{address}/transactions?limit={limit}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            transactions = response.json()
            logger.info(f"Fetched {len(transactions)} transactions")
            return transactions
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Failed to fetch transactions after max retries")
                return []

def fetch_pool_data(api_url: str, pool_address: str) -> Dict:
    """Fetch liquidity pool data."""
    try:
        url = f"{api_url}/accounts/{pool_address}/resource/0x1::coin::CoinStore"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch pool data: {str(e)}")
        return {}
