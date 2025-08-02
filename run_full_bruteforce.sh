uv run trainer.py -cn simple_siglip_only_nitc_mvs \
    dataset.dataset_name='CUHK-PEDES' \
    trainer.max_epochs=200 \
    dataset.batch_size=48 \
    loss.SS=true \
    loss.ss_loss_weight=0.4 \
    loss.CITC=true \
    loss.citc_loss_weight=0.1 \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    +early_stopping.monitor=val_score \
    +early_stopping.patience=10 \
    +early_stopping.mode=max \
    +early_stopping.verbose=true \
    img_size_st="'(256,256)'"