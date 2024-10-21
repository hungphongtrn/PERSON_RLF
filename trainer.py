import logging
import os

import hydra
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import OmegaConf, DictConfig

from lightning_models import LitTBPS
from lightning_data import TBPSDataModule
from utils.logger import setup_logging
from utils.visualize_test import visualize_test


# Setting up the loggeer
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)


@hydra.main(version_base=None, config_path="config")
def run(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    # Set the seed
    seed_everything(config.seed)

    training_logger, checkpoint_callback = setup_logging(config)

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
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Preparing the model
    model = LitTBPS(
        config,
        vocab_size=tokenizer.true_vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        num_iters_per_epoch=len(train_loader),
        num_classes=dm.num_classes,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Preparing the trainer
    trainer_args = config.trainer
    logging.info(f"Trainer Args: {trainer_args}")
    logging.info(f"CE Loss ignored tokens: {dm.tokenizer.pad_token_id}")
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        logger=training_logger,
        **trainer_args,
    )
    logging.info(f"Test loader length: {len(iter(test_loader))}")

    if config.logger.logger_type == "wandb":
        training_logger.watch(model, log="all")

    trainer.validate(model, val_loader)

    if config.get("ckpt_path", None):
        logging.info(f"Resuming from checkpoint: {config.ckpt_path}")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=config.ckpt_path,
        )
    else:
        logging.info("Starting training from scratch")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(model, ckpt_path="best", dataloaders=test_loader)
    fig = visualize_test(model.test_final_outputs, tokenizer)
    plt.savefig(os.path.join(training_logger.save_dir, "test_visualization.png"))

    if config.logger.logger_type == "wandb":
        # Log figure to wandb
        training_logger.log({"test_visualization": fig})


if __name__ == "__main__":
    run()
