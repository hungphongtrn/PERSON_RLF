apt-get install byobu
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
gdown 1aZ2355gZpTH-tdrdHl5Y0PDnX8MnaV1Z # download dataset
unzip CUHK-PEDES.zip
cd CUHK-PEDES
gdown 12RZMdUpH2u5lX4s78kKwBphKBUpO2ZRt # download reid_raw.json
uv run trainer.py -cn simple_siglip_only_nitc_mvs img_size_st="'(256,256)'" trainer.max_epochs=60 loss.SS=true loss.CITC=true dataset=cuhk_pedes