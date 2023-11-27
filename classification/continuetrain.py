from classification import train
from argparse import Namespace, ArgumentParser
from datetime import datetime
import json
import logging
from pathlib import Path

import yaml
import torch
import torchvision
import pytorch_lightning as pl
import pandas as pd

from classification import utils_global

def main():
    logging.basicConfig(level=logging.INFO, filename="/work3/s212495/2trainres.log")
    logger = pl.loggers.TensorBoardLogger(save_dir="/work3/s212495/tblog", name="resnetlog")
    out_dir = Path("/work3/s212495/data/models/base_Mwith4gpu/") / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init 
    model = train.resnetregressor.load_from_checkpoint(
        checkpoint_path = "/work3/s212495/data/models/base_Mwith4gpu/231120-2349/ckpts/epoch=14-the_val_loss=4003.64.ckpt")

    checkpoint_dir = out_dir / "ckpts" 
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                filename='{epoch}-{the_val_loss:.2f}',
                                                save_top_k = 10,
                                                monitor = 'the_val_loss', 
                                                mode = 'min')



    trainer = pl.Trainer(
        max_epochs=100,
        precision=16,
        num_nodes=1,
        gradient_clip_val=0.8,
        reload_dataloaders_every_n_epochs=1,
        logger=logger,
        accelerator="gpu",
        devices=-1,
        val_check_interval=4000, 
        callbacks=[checkpointer],
        enable_progress_bar = False
    )
    
    trainer.fit(model, 
                ckpt_path="/work3/s212495/data/models/base_Mwith4gpu/231120-2349/ckpts/epoch=14-the_val_loss=4003.64.ckpt")

    
if __name__ == "__main__":
    main()

