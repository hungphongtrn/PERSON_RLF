from omegaconf import OmegaConf
from loguru import logger
import lightning as L

from lighning_models import LitIRRA
from lightning_data import IRRADataModule

config = OmegaConf.load("config.yaml")
dm = IRRADataModule(config)
dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()
model = LitIRRA(config, img_loader=test_loader[0], text_loader=test_loader[1], num_classes=dm.num_classes)
trainer = L.Trainer(fast_dev_run=1, precision="16-mixed", logger=logger, num_sanity_val_steps=2, profiler="simple")
trainer.fit(model, train_dataloaders=train_loader)