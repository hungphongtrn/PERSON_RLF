apt-get install byobu
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run gdown 1aZ2355gZpTH-tdrdHl5Y0PDnX8MnaV1Z # download dataset
unzip CUHK-PEDES.zip
cd CUHK-PEDES
uv run gdown 12RZMdUpH2u5lX4s78kKwBphKBUpO2ZRt # download reid_raw.json