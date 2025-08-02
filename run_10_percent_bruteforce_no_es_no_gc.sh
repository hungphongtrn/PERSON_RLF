uv run trainer.py -cn simple_siglip_only_nitc_mvs \
    dataset.dataset_name='CUHK-PEDES' \
    dataset.batch_size=64 \
    dataset.proportion=0.1 \
    trainer.max_epochs=200 \
    loss.SS=true \
    loss.ss_loss_weight=0.4 \
    loss.CITC=true \
    loss.citc_loss_weight=0.1
