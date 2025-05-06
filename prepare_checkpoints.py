from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip-base-patch16-256-multilingual",
    local_dir="m_siglip_checkpoints",
    max_workers=4,
)
