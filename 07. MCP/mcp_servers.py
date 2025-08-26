import os
from dotenv import load_dotenv

load_dotenv()

SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")
MCP_SERVERS_CONFIG = {
    "yfinance": {
        "url": f"https://server.smithery.ai/@imbenrabi/financial-modeling-prep-mcp-server/mcp?api_key={SMITHERY_API_KEY}",
        "transport": "streamable_http",
    },
}