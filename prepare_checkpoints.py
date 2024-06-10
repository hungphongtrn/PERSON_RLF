from huggingface_hub import snapshot_download

snapshot_download(repo_id="google/siglip-base-patch16-256-multilingual",
                  local_dir="siglip_checkpoints",
                  local_dir_use_symlinks=False)
