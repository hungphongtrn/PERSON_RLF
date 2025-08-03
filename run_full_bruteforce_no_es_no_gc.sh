uv run trainer.py -cn simple_siglip_only_nitc_mvs \
    dataset="cuhk_pedes" \
    dataset.dataset_name='CUHK-PEDES' \
    trainer.max_epochs=200 \
    dataset.batch_size=48 \
    loss.SS=true \
    loss.ss_loss_weight=0.4 \
    loss.CITC=true \
    loss.citc_loss_weight=0.1 \
    img_size_st="'(256,256)'"
