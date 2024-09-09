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
