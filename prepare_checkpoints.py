from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip-base-patch16-256-multilingual",
    local_dir="siglip_checkpoints",
    max_workers=4,
)

# snapshot_download(
#     repo_id="openai/clip-vit-base-patch16", local_dir="clip_checkpoints", max_workers=2
# )
