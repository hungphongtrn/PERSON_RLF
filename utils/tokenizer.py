from transformers import AutoTokenizer


def get_tokenizer(args):
    backbone_path = args.backbone_path
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_path, model_max_length=args.text_length
    )
    # Use token with id 250000 as the mask token
    mask_token = tokenizer.convert_ids_to_tokens(250000)
    tokenizer.add_special_tokens({"mask_token": mask_token})
    return tokenizer
