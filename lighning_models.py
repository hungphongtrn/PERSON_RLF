import lightning as L
import torch

from model.build import IRRA
from solver import build_lr_scheduler, build_optimizer
from utils.metrics import Evaluator


class LitIRRA(L.LightningModule):
    def __init__(self, args, img_loader, text_loader, num_classes=11003):
        super().__init__()

        self.model = IRRA(args, num_classes=num_classes)
        self.current_task = self.model.current_task
        self.evaluator = Evaluator(img_loader, text_loader)

        self.args = args

    def training_step(self, batch, batch_idx):
        ret = self.model(batch)
        loss = sum([ret[k] for k in ret if k.endswith("loss")])

        # Logging and show in progrewss bar
        self.log_dict(ret, on_step=True, on_epoch=True, prog_bar=True)
        self.log("total_loss", loss, on_epoch=True, prog_bar=True)

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
        optimizer = build_optimizer(self.args, self.model)
        scheduler = build_lr_scheduler(self.args, optimizer)
        return [optimizer], [scheduler]
