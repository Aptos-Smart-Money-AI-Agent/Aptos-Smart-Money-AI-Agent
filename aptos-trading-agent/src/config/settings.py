import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
PRIVATE_KEY = os.getenv("APTOS_PRIVATE_KEY")
NETWORK = os.getenv("APTOS_NETWORK", "testnet")

# API endpoints for Aptos networks
API_ENDPOINTS = {
    "mainnet": "https://fullnode.mainnet.aptoslabs.com/v1",
    "testnet": "https://fullnode.testnet.aptoslabs.com/v1",
}

# Token configurations
TOKEN_CONFIGS = {
    "APT": {
        "module_address": "0x1",
        "module_name": "aptos_coin",
        "struct_name": "AptosCoin"
    },
    # Add other tokens as needed
}
