import logging

import hydra
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, DictConfig

from lightning_models import LitTBPS
from lightning_data import TBPSDataModule


# Setting up the loggeer
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    # Set the seed
    seed_everything(config.seed)

    # Get the version
    version = config.get("version", None)

    # Modify the config if use MLM
    if config.loss.MLM:
        config.tokenizer.vocab_size += 1
        config.tokenizer.add_mask_token = True
        config.backbone.text_config.vocab_size = config.tokenizer.vocab_size

    # Load the data module
    dm = TBPSDataModule(config)
    dm.setup()

    tokenizer = dm.tokenizer

    # Log an example of the dataset
    logging.info(f"Example of the dataset: {dm.train_set[0]}")

    # Prepare dataloader
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    # Experiment name
    model_short_name = config.backbone.type.split(".")[-1]
    experiment_name = config.dataset.dataset_name + "_" + model_short_name

    # Preparing Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        filename=experiment_name + "_{epoch}-{total_loss:.2f}-{eval_score:.2f}",
        save_top_k=3,
        monitor="eval_score",
        mode="max",
        enable_version_counter=True,
    )

    # Preparing the model
    model = LitTBPS(
        config,
        vocab_size=tokenizer.true_vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        num_iters_per_epoch=len(train_loader),
        num_classes=dm.num_classes,
    )

    # Preparing the monitors
    board_logger = TensorBoardLogger(
        save_dir=config.trainer.default_root_dir,
        version=version,
        name=experiment_name,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Preparing the trainer
    trainer_args = config.trainer
    logging.info(f"Trainer Args: {trainer_args}")
    logging.info(f"CE Loss ignored tokens: {dm.tokenizer.pad_token_id}")
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        logger=board_logger,
        **trainer_args,
    )
    logging.info(f"Test loader length: {len(iter(test_loader))}")

    trainer.validate(model, test_loader)

    if config.get("ckpt_path", None):
        logging.info(f"Resuming from checkpoint: {config.trainer.ckpt_path}")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
            ckpt_path=config.ckpt_path,
        )
    else:
        logging.info("Starting training from scratch")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    run()
