#!/bin/bash


python medaf_lightning_trainer.py --mode test \
    --checkpoint checkpoints/medaf_lightning/2025-10-14-17_15/last.ckpt \
    --config config_lightning.yaml


# python medaf_lightning_trainer.py --mode train \
    # --resume checkpoints/medaf_lightning/medaf-lightning-epoch=10-val_loss=0.0000.ckpt

# python medaf_lightning_trainer.py --mode train \
#     --resume checkpoints/medaf_lightning/2025-10-14-15_48/last.ckpt \
#     --config config_lightning.yaml