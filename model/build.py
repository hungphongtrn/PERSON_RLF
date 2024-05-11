from collections import OrderedDict
import os

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm
from .clip_vision import build_CLIPVision_from_openai_pretrained


def load_sentence_transformers():
    MODEL_CP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sentence_transformers_cp")
    model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1", device="cpu", cache_folder=MODEL_CP_PATH)
    return model


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.image_model, base_cfg = build_CLIPVision_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.text_model = load_sentence_transformers()
        # from text model word embedding (sequence_length, embed_dim) to (sequence_length, embed_dim)
        self.proj_text = nn.Linear(768, base_cfg['embed_dim'])  # TODO: get the embed_dim from the model
        self.embed_dim = base_cfg['embed_dim']

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
            nn.init.normal_(self.proj_text.weight, std=proj_std)
            nn.init.normal_(self.proj_text.bias, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.image_model(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        # text will be a dict that has forms {"input_ids": ..., "attention_mask": ...}
        x = self.text_model.forward(text)['sentence_embedding']
        return x

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats = self.image_model(images)

        i_feats = image_feats[:, 0, :].float()  # extract the first feature for each image
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = self.text_model.forward(caption_ids)['sentence_embedding']

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
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.text_model.forward(mlm_ids)['token_embeddings']
            down_scaled_mlm_feats = self.proj_text(mlm_feats)

            x = self.cross_former(down_scaled_mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)  # [batch_size * text_len]
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            # return the idx that is not -100 but smaller than 77
            mlm_label_idx = torch.nonzero(mlm_labels != -100)
            mlm_label_idx = mlm_label_idx[mlm_label_idx < 77]
            # print("mlm_label_idx", mlm_label_idx)
            # mlm_label_idx = torch.nonzero(mlm_labels)
            # print("pred[mlm_label_idx]", pred[mlm_label_idx])
            # print("mlm_labels[mlm_label_idx]", mlm_labels[mlm_label_idx])
            
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    return model
