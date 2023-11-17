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
from classification.dataset import MsgPackIterableDatasetMultiTargetWithDynLabels


class resnetregressor(pl.LightningModule):
    def __init__(self, modelparams: Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.model, self.regressor = self.__build_model()
        self.validation_step_outputs = []

 

    def __build_model(self):
        logging.info("Build model")
        model, nfeatures = utils_global.build_base_model(self.hparams.modelparams.arch)

        regressor = torch.nn.Sequential(
            torch.nn.Linear(nfeatures, 64),  # 64是你选择的隐藏层大小
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Tanh()# 输出两个数字（-1 - 1）
        )

        if self.hparams.modelparams.weights:
            logging.info("Load weights from pre-trained model")
            model, regressor = utils_global.load_weights_if_available(
                model, regressor, self.hparams.modelparams.weights
            )

        return model, regressor

    def forward(self, x):
        fv = self.model(x)
        yhats = self.regressor(fv)
        return yhats
    
 
        

    def training_step(self, batch, batch_idx):
        images, target = batch
        
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output = self(images)  #形状为 (batch_size, 2) 
        
        
       # 缩放和映射
        output_scaled = torch.stack([
            output[:, 0] * 90.0,   # 映射到 -90 到 +90 范围
            output[:, 1] * 180.0   # 映射到 -180 到 +180 范围
        ], dim=1)

        # 检测 output 中是否存在 NaN 值
        has_nan_output = torch.isnan(output).any().item()
        if has_nan_output:
            print("There is nan in trainoutput")
        # 检测 output_scaled 中是否存在 NaN 值
        has_nan_output_scaled = torch.isnan(output_scaled).any().item()
        if has_nan_output_scaled:
            print("There is nan in trainoutput_scaled")
        
        losses = [
            utils_global.vectorized_gc_distance(output_scaled[i][0],output_scaled[i][1], target[0][i],target[1][i])
            for i in range(output.shape[0])
        ]
        loss = sum(losses)
        errors = [loss.item() for loss in losses]
        self.log("train_loss", loss)
        output = {
            "loss" : loss,
            "losses" : errors}
        return output

    def on_train_batch_end(self,outputs, batch, batch_idx):
        if batch_idx % 3999 == 0:
            print("----------------train_batch_end_loss_every4000---------------")
            print(outputs["losses"])
            print("---------------------------------------------------")
    
    
    def validation_step(self, batch, batch_idx):
        images, target = batch #iamge是（batch size，2），target是两个张量的列表，一个是lat，一个是lon
        
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self(images)
        # 缩放和映射
        output_scaled = torch.stack([
            output[:, 0] * 90.0,   # 映射到 -90 到 +90 范围
            output[:, 1] * 180.0   # 映射到 -180 到 +180 范围
        ], dim=1)
        
        
        
        
        
        has_nan_output = torch.isnan(output).any().item()
        if has_nan_output:
            print("There is nan in valoutput")
        # 检测 output_scaled 中是否存在 NaN 值
        has_nan_output_scaled = torch.isnan(output_scaled).any().item()
        if has_nan_output_scaled:
            print("There is nan in valoutput_scaled")

        # loss calculation
        losses = [
            utils_global.vectorized_gc_distance(output_scaled[i][0],output_scaled[i][1], target[0][i],target[1][i])
            for i in range(output.shape[0])
        ]
       
        loss = sum(losses)

        thissize = output.shape[0]
        # 计算误差统计
        errors = [loss.item() for loss in losses]
    # 统计不同误差范围内的样本数量
        num_samples = len(errors)
        error_100 = sum([1 for error in errors if error <= 100])
        error_500 = sum([1 for error in errors if  error <= 500])
        error_1000 = sum([1 for error in errors if  error <= 1000])
        error_2000 = sum([1 for error in errors if  error <= 2000])

    # 输出统计信息
#         logging.info("NumSamples", num_samples)
#         logging.info("Error100", error_100)
#         logging.info("Error500", error_500)
#         logging.info("Error1000", error_1000)
#         logging.info("Error2000", error_2000)
       
     
       

        output = {
            "val_loss": loss,
            "errors" : errors,
            "avg_1loss":loss/thissize, 
            "size" : thissize,
            "ACC100" : error_100,
            "ACC500" : error_500,
            "ACC1000" : error_1000,
            "ACC2000" : error_2000
        }
        self.log("val_loss", loss)
        # print("-------------valoutput----------")
        # print(output)
        # print("------------------------------------------------------")
        self.validation_step_outputs.append(output)
        return output
    
    
    
    
    
    def on_validation_batch_end(self,outputs, batch, batch_idx): 
        if batch_idx % 100 == 0:
            print("----------------val_batch_end_loss---------------")
            print(outputs["errors"])
            print("---------------------------------------------------")
    
        
            
           
        
        
        
        
        
    
    def on_validation_epoch_end(self):
        epoch_num = self.current_epoch
        logging.info(f"Starting epoch {epoch_num}")
        avg_loss = torch.tensor([x["avg_1loss"].item() for x in self.validation_step_outputs]).mean()

    
        total_samples = sum([x["size"] for x in self.validation_step_outputs])
        total_error_100 = sum([x["ACC100"] for x in self.validation_step_outputs])
        total_error_500 = sum([x["ACC500"] for x in self.validation_step_outputs])
        total_error_1000 = sum([x["ACC1000"] for x in self.validation_step_outputs])
        total_error_2000 = sum([x["ACC2000"] for x in self.validation_step_outputs])
    
        error_100_ratio = total_error_100 / total_samples
        error_500_ratio = total_error_500 / total_samples
        error_1000_ratio = total_error_1000 / total_samples
        error_2000_ratio = total_error_2000 / total_samples
    
        logging.info("the_val_loss: %s", avg_loss.item())
        logging.info("100_accratio: %s", error_100_ratio)
        logging.info("500_accratio: %s", error_500_ratio)
        logging.info("1000_accratio: %s", error_1000_ratio)
        logging.info("2000_accratio: %s", error_2000_ratio)
        self.log("the_val_loss", avg_loss)
        self.log("100_accratio", error_100_ratio)
        self.log("500_accratio", error_500_ratio)
        self.log("1000_accratio", error_1000_ratio)
        self.log("2000_accratio", error_2000_ratio)
        self.validation_step_outputs.clear()


    def _multi_crop_inference(self, batch):
        images, meta_batch = batch
        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape crop dimension to batch
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        # forward pass
        yhats = self(images)
        yhats = [torch.nn.functional.softmax(yhat, dim=1) for yhat in yhats]

        # respape back to access individual crops
        yhats = [
            torch.reshape(yhat, (cur_batch_size, ncrops, *list(yhat.shape[1:])))
            for yhat in yhats
        ]

        # calculate max over crops
        yhats = [torch.max(yhat, dim=1)[0] for yhat in yhats]

        hierarchy_preds = None
        if self.hierarchy is not None:
            hierarchy_logits = torch.stack(
                [yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)],
                dim=-1,
            )
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        return yhats, meta_batch, hierarchy_preds

    def inference(self, batch):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        pred_class_dict = {}
        pred_lat_dict = {}
        pred_lng_dict = {}
        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            pred_lat_dict[pname] = pred_lats
            pred_lng_dict[pname] = pred_lngs
            pred_class_dict[pname] = pred_classes

        return meta_batch["img_path"], pred_class_dict, pred_lat_dict, pred_lng_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        yhats, meta_batch, hierarchy_preds = self._multi_crop_inference(batch)

        distances_dict = {}
        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)

            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                meta_batch["latitude"].type_as(pred_lats),
                meta_batch["longitude"].type_as(pred_lngs),
            )
            distances_dict[pname] = distances

        return distances_dict

    def on_test_epoch_end(self, outputs):
        result = utils_global.summarize_test_gcd(
            [p.shortname for p in self.partitionings], outputs, self.hierarchy
        )
        return {**result}

    def configure_optimizers(self):

        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.modelparams.optim["params"]
        )
        Ascheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim_feature_extrator, **self.hparams.modelparams.scheduler["params"]
        )
        
        return [optim_feature_extrator],[Ascheduler]
        # return {
        #     "optimizer": optim_feature_extrator,
        #     "lr_scheduler": {
        #         "scheduler": torch.optim.lr_scheduler.MultiStepLR(
        #             optim_feature_extrator, **self.hparams.modelparams.scheduler["params"]
        #         ),
        #         "interval": "epoch",
        #         "name": "lr"
        #     },
        # }

    def train_dataloader(self):

        with open(self.hparams.modelparams.after_train_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.modelparams.msgpack_train_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.modelparams.key_img_id,
            key_img_encoded=self.hparams.modelparams.key_img_encoded,
            shuffle=True,
            transformation=tfm,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.modelparams.batch_size,
            num_workers=self.hparams.modelparams.num_workers_per_loader,
            pin_memory=True,
        )
        print("-------------traindataloader lenth---------------")
        print(len(dataloader))
        print("-------------------------------------------------")
        return dataloader

    def val_dataloader(self):

        with open(self.hparams.modelparams.after_val_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.modelparams.msgpack_val_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.modelparams.key_img_id,
            key_img_encoded=self.hparams.modelparams.key_img_encoded,
            shuffle=False,
            transformation=tfm,
            cache_size=1024,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.modelparams.batch_size,
            num_workers=self.hparams.modelparams.num_workers_per_loader,
            pin_memory=True,
        )
        print("-------------valdataloader lenth---------------")
        print(len(dataloader))
        print("-------------------------------------------------")

        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("config/newbaseM.yml"))
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, filename="/work3/s212495/trainres.log")
    logger = pl.loggers.CSVLogger(save_dir="/work3/s212495/csvlog", name="resnetlog")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    utils_global.check_is_valid_torchvision_architecture(model_params["arch"])

    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init 
    model = resnetregressor(modelparams=Namespace(**model_params))

    checkpoint_dir = out_dir / "ckpts" 
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                filename='{epoch}-{the_val_loss:.2f}',
                                                save_top_k = 10,
                                                monitor = 'the_val_loss', 
                                                mode = 'min')

    progress_bar_refresh_rate = False
    if args.progbar:
        progress_bar_refresh_rate = True

    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        accelerator="gpu",
        devices=-1,
        val_check_interval=model_params["val_check_interval"], 
        callbacks=[checkpointer],
        enable_progress_bar=progress_bar_refresh_rate,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
