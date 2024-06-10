import lightning as L

from model.build import IRRA
from utils.metrics import Evaluator
from solver import build_optimizer, build_lr_scheduler

class LitIRRA(L.LightningModule):
    def __init__(self, args, img_loader, text_loader, num_classes=11003):
        super().__init__()
        
        self.model = IRRA(args, num_classes=num_classes)
        self.current_task = self.model.current_task
        self.evaluator = Evaluator(img_loader, text_loader)
        
        self.args = args
        
    def training_step(self, batch, batch_idx):
        ret = self.model(batch)
        loss = sum([ret[k] for k in ret if k.endswith('loss')])
        
        # Logging and show in progrewss bar
        self.log_dict(ret, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self, outputs):
        # Run the evaluator on the entire dataset
        eval_score = self.evaluator.eval(self.model)

        # Log evaluator results
        self.log('eval_score', eval_score, prog_bar=True)
        return eval_score
        
    def configure_optimizers(self): 
        optimizer = build_optimizer(self.args, self.model)
        scheduler = build_lr_scheduler(self.args, optimizer)
        return [optimizer], [scheduler]

        
        
        
        