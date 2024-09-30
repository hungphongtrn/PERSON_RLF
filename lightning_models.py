import logging

import lightning as L
import torch
from lightning.pytorch.utilities import grad_norm

from model.build import build_backbone_with_proper_layer_resize
from model.tbps import TBPS
from solver import build_lr_scheduler, build_optimizer
from utils.metrics import Evaluator


class LitTBPS(L.LightningModule):
    def __init__(
        self,
        config,
        img_loader,
        text_loader,
        tokenizer,
        train_loader_size,
        num_classes=11003,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.backbone = build_backbone_with_proper_layer_resize(self.config.backbone)
        self.model = TBPS(
            config=config,
            backbone=self.backbone,
            tokenizer=tokenizer,
            num_classes=num_classes,
        )
        self.evaluator = Evaluator(img_loader, text_loader)
        self.train_loader_size = train_loader_size

    def training_step(self, batch, batch_idx):
        # Get the current epoch
        epoch = self.trainer.current_epoch
        step = self.trainer.global_step
        alpha = self.config.experiment.softlabel_ratio
        if epoch == 0:
            alpha *= min(1.0, step / self.train_loader_size)

        ret = self.model(batch, alpha)
        loss = sum([ret[k] for k in ret if k.endswith("loss")])

        # Logging and show in progress bar
        self.log_dict(ret, on_step=True, on_epoch=True, prog_bar=False)
        self.log("softlabel_ratio", alpha, on_step=True, on_epoch=False, prog_bar=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Run the evaluator on the entire dataset
        with torch.no_grad():
            eval_score = self.evaluator.eval(self.model)
        # eval_score = self.evaluator.eval(self.model)
        torch.cuda.empty_cache()
        # Log evaluator results
        self.log("eval_score", eval_score)
        return eval_score

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config.optimizer, self.model)
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
