from transformers import AutoTokenizer


def get_tokenizer(args):
    backbone_path = args.backbone_path
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, model_max_length=args.text_length)
    # Add mask token to the tokenizer
    tokenizer.get_vocab().update({"<mask>": 250000})
    tokenizer.add_special_tokens({'mask_token': '<mask>'})
    assert tokenizer.mask_token == '<mask>', "Failed to add mask token to the tokenizer"
    assert tokenizer.mask_token_id == 250000, "Mask token id is not 250000"
    return tokenizer