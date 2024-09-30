# Person-Search using SigLIP

## SETUP

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

## Logs

- 13/9/24:
    - NaN loss is caused by `RuntimeError: Function 'SoftmaxBackward0' returned nan values in its 0th output.` in `attn_weights = nn.functional.softmax`.
        - Gradient norm is too high:
            - [X] Reduce the learning rate (1e-5->3e-6)
            - [X] Clip the gradient
    - The error was caused by the tokenizer. The original tokenizer includes `extra_tokens_...`, which makes the number of tokens larger than the embedding size. This causes the model to return NaN values.


- 10/9/24: Successfully ran 30% of the first epoch but encounter an error related to CUDA which might be due to NAN values in the loss. Next steps:
    - Investigate what caused NaN values in the loss
