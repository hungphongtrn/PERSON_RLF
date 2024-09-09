import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from lighning_models import LitIRRA
from lightning_data import IRRADataModule

# Loading the configuration file
seed_everything(42)
config = OmegaConf.load("config.yaml")

# Loading the data module
dm = IRRADataModule(config)
dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

# Preparing Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    filename="VN3k-mixed-ENVI-{epoch}-{total_loss:.2f}-{eval_score:.2f}",
    save_top_k=10,
    monitor="eval_score",
    mode="max",
    enable_version_counter=True,
)

# Preparing the model
model = LitIRRA(
    config,
    img_loader=test_loader[0],
    text_loader=test_loader[1],
    num_classes=dm.num_classes,
)

# Preparing the trainer
logger = TensorBoardLogger(
    save_dir="logs/", version=1, name="vn3k_mixed_envi_train_and_test"
)
trainer = L.Trainer(
    precision="16-mixed",
    callbacks=[checkpoint_callback],
    max_epochs=config.num_epochs,
    logger=logger,
    default_root_dir="logs/",
    enable_progress_bar=True,
    enable_model_summary=True,
)
trainer.fit(model, train_dataloaders=train_loader)
