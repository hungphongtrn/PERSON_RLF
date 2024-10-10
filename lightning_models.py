import logging

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm
from prettytable import PrettyTable

from model.build import build_backbone_with_proper_layer_resize
from model.tbps import TBPS
from solver import build_lr_scheduler, build_optimizer
from utils.metrics import rank

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class LitTBPS(L.LightningModule):
    def __init__(
        self,
        config,
        vocab_size,
        pad_token_id,
        num_iters_per_epoch,
        num_classes=11003,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.backbone = build_backbone_with_proper_layer_resize(self.config.backbone)
        self.model = TBPS(
            config=config,
            backbone=self.backbone,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_classes=num_classes,
        )
        self.num_iters_per_epoch = num_iters_per_epoch
        self.validation_step_outputs = {
            "text_ids": [],
            "image_ids": [],
            "text_feats": [],
            "image_feats": [],
        }

    def training_step(self, batch, batch_idx):
        # Get the current epoch
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step
        alpha = self.config.experiment.softlabel_ratio
        if epoch == 0:
            alpha *= min(1.0, step / self.num_iters_per_epoch)

        ret = self.model(batch, alpha)
        loss = sum([ret[k] for k in ret if k.endswith("loss")])

        # Logging and show in progress bar
        self.log_dict(ret, on_step=True, on_epoch=True, prog_bar=False)
        self.log("softlabel_ratio", alpha, on_step=True, on_epoch=False, prog_bar=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch["img"]:
            pid = batch["img"]["pids"]
            img = batch["img"]["images"]
            img_feat = self.model.encode_image(img)
            self.validation_step_outputs["image_ids"].append(pid.view(-1))  # flatten
            self.validation_step_outputs["image_feats"].append(img_feat)

        if batch["txt"]:
            pid = batch["txt"]["pids"]
            caption_input = {
                "input_ids": batch["txt"]["caption_input_ids"],
                "attention_mask": batch["txt"]["caption_attention_mask"],
            }
            text_feat = self.model.encode_text(caption_input)
            self.validation_step_outputs["text_ids"].append(pid.view(-1))  # flatten
            self.validation_step_outputs["text_feats"].append(text_feat)

    def on_validation_epoch_end(self):
        logging.info("Validation epoch end")
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])

        image_ids = torch.cat(self.validation_step_outputs["image_ids"], 0)
        image_feats = torch.cat(self.validation_step_outputs["image_feats"], 0)
        text_ids = torch.cat(self.validation_step_outputs["text_ids"], 0)
        text_feats = torch.cat(self.validation_step_outputs["text_feats"], 0)

        # Logging statistics of the validation set
        logging.info(f"Image ids: {image_ids.size()}")
        logging.info(f"Text ids: {text_ids.size()}")
        logging.info(f"Image feats: {image_feats.size()}")
        logging.info(f"Text feats: {text_feats.size()}")

        similarity = text_feats @ image_feats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity=similarity,
            q_pids=text_ids,
            g_pids=image_ids,
            max_rank=10,
            get_mAP=True,
        )
        t2i_cmc, t2i_mAP, t2i_mINP = (
            t2i_cmc.tolist(),
            t2i_mAP.tolist(),
            t2i_mINP.tolist(),
        )
        table.add_row(["t2i", t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
            similarity=similarity.t(),
            q_pids=image_ids,
            g_pids=text_ids,
            max_rank=10,
            get_mAP=True,
        )
        i2t_cmc, i2t_mAP, i2t_mINP = (
            i2t_cmc.tolist(),
            i2t_mAP.tolist(),
            i2t_mINP.tolist(),
        )
        table.add_row(["i2t", i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        results = {
            "t2i_R1": t2i_cmc[0],
            "t2i_R5": t2i_cmc[4],
            "t2i_R10": t2i_cmc[9],
            "t2i_mAP": t2i_mAP,
            "t2i_mINP": t2i_mINP,
            "i2t_R1": i2t_cmc[0],
            "i2t_R5": i2t_cmc[4],
            "i2t_R10": i2t_cmc[9],
            "i2t_mAP": i2t_mAP,
            "i2t_mINP": i2t_mINP,
        }

        self.log_dict(results, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "eval_score", results["t2i_R1"], on_step=False, on_epoch=True, prog_bar=True
        )
        logging.info("\n" + str(table))

        # Reset the outputs
        self.validation_step_outputs = {
            "text_ids": [],
            "image_ids": [],
            "text_feats": [],
            "image_feats": [],
        }

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config.optimizer, self.model)
        self.config.scheduler.n_iter_per_epoch = self.num_iters_per_epoch
        scheduler = build_lr_scheduler(self.config.scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        all_norms = norms[f"grad_{float(2)}_norm_total"]
        self.log("grad_norm", all_norms, on_step=True, on_epoch=True, prog_bar=True)
