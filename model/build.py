import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from model import objectives
from model.layers import Transformer, QuickGELU, LayerNorm
from model.siglip.modeling_siglip import SiglipTextModel, SiglipVisionModel

logger = logging.getLogger(__name__)

class IRRA(nn.Module):
    def __init__(self, 
                 args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.current_task = self.args.loss_names
        logger.info(f'Training Model with {self.current_task} tasks')
        
        self.text_model = SiglipTextModel.from_pretrained(args.backbone_path)
        # Resize the token and positional embeddings for training
        if args.training:
            self.text_model.resize_token_embeddings(args.vocab_size)
        # self.text_model.resize_position_embeddings(args.text_length)
        self.image_model = SiglipVisionModel.from_pretrained(args.backbone_path)
        self.embed_dim = self.text_model.config.hidden_size
        logger.info(f'Embedding Dimension: {self.embed_dim}')
        self.interpolate_pos_encoding = args.interpolate_pos_encoding

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        """Encodes an image into a tensor.
        Args:
            image (PIL.Image): Image to be encoded.
        Returns:
            torch.Tensor: The encoded image, use pooler_output as the image embedding."""
        x = self.image_model(pixel_values=image, interpolate_pos_encoding=self.interpolate_pos_encoding).pooler_output
        return x

    def encode_text(self, text):
        """Encodes text into a tensor.
        Args:
            text (dict): Text to be encoded, containing the keys "input_ids" and "attention_mask".
        Returns:
            torch.Tensor: The encoded text, use pooler_output as the sentence embedding."""
        # text will be a dict that has forms {"input_ids": ..., "attention_mask": ...}
        x = self.text_model(**text).pooler_output
        return x

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_input = batch['caption_input']
        image_model_output = self.image_model(images, interpolate_pos_encoding=self.interpolate_pos_encoding)
        text_model_output = self.text_model(**caption_input)
        
        i_feats = image_model_output.pooler_output  # image embedding
        t_feats = text_model_output.pooler_output   # text embedding

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats).float()
            text_logits = self.classifier(t_feats).float()
            ret.update({'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_inputs = batch['mlm_inputs']

            image_feats = image_model_output.last_hidden_state   # image features

            mlm_feats = self.text_model.forward(**mlm_inputs).last_hidden_state  # masked text features

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  

            scores = x.reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)  # [batch_size * text_len]
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels != 1)

            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret
