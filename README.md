# SETUP

```bash
pip install -r requirements.txt
```

- Get the siglip checkpoint

```bash
python prepare_checkpoints.py
```

- Modify the tokenizer_config.json by adding the following line to the `added_tokens_decoder`

```json
{
    "250000": {
      "content": "<mask>",
      "lstrip": true,
      "normalized": false,
      "rstrip": true,
      "single_word": false,
      "special": true
    }
}
```

# Logs

- 10/9/24: Successfully ran 30% of the first epoch but encounter an error related to CUDA which might be due to NAN values in the loss. Next steps:
    - Investigate what caused NaN values in the loss