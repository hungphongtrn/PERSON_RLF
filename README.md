# Person-Search using SigLIP

## Setup

1. Clone the repository
```bash
git clone https://github.com/hungphongtrn/PERSON_RLF.git
```

2. Install uv package manager and sync the dependencies
```bash
cd PERSON_RLF
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

3. Download the `siglip-base-patch16-256-multilingual` checkpoints
```bash
uv run prepare_checkpoints.py
```

4. Put the CUHK-FULL dataset in the root folder
Here is the sample structure of the project
```bash
.
|-- clip_checkpoints
|-- config
|-- CUHK-PEDES          # This is the dataset folder for CUHK-PEDES
|-- VN3K                # This is the dataset folder for VN3K
|-- data
|-- experiments
|-- lightning_data.py
|-- lightning_models.py
|-- model
|-- m_siglip_checkpoints
|-- outputs
|-- prepare_checkpoints.py
|-- __pycache__
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- run.sh
|-- siglip_checkpoints
|-- solver
|-- trainer.py
|-- utils
|-- uv.lock
|-- ...
```

5. Log in to the Weights & Biases
```bash
uv run wandb login <API_KEY>
```

### Run the experiments

1. CUHK-FULL dataset
```bash
# With m-SigLIP
# Run the training with TBPS method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=random loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=tbps_clip_no_decay optimizer.param_groups.default.lr=1e-5
# Run the training with IRRA method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=identity dataset.num_instance=1 loss=irra loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=irra_no_decay optimizer.param_groups.default.lr=1e-5
```
