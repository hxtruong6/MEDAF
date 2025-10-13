#!/bin/bash


# python medaf_lightning_trainer.py --mode test --checkpoint checkpoints/medaf_lightning/medaf-lightning-epoch=10-val_loss=0.0000.ckpt

python medaf_lightning_trainer.py --mode train \
    --resume checkpoints/medaf_lightning/medaf-lightning-epoch=10-val_loss=0.0000.ckpt
