from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
import timm 
import os
from pytorch_lightning import LightningModule,  seed_everything
seed_everything(7)


class CifarLitModule(LightningModule):
    def __init__(self, model_name, optimizer:torch.optim.Optimizer):
        super().__init__()

        self.save_hyperparameters()
        # self.model = create_model()
        self.model = timm.create_model(model_name, pretrained=True, num_classes = 10)
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)
        # return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(
    #         self.parameters(),
    #         lr=self.hparams.lr,
    #         momentum=0.9,
    #         weight_decay=5e-4,
    #     )
    #     steps_per_epoch = 45000 // BATCH_SIZE
    #     scheduler_dict = {
    #         "scheduler": OneCycleLR(
    #             optimizer,
    #             0.1,
    #             epochs=self.trainer.max_epochs,
    #             steps_per_epoch=steps_per_epoch,
    #         ),
    #         "interval": "step",
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Perform a forward pass through the model `self.net`.

    #     :param x: A tensor of images.
    #     :return: A tensor of logits.
    #     """
    #     return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # def training_step(
    #     self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    # ) -> torch.Tensor:
    #     """Perform a single training step on a batch of data from the training set.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target
    #         labels.
    #     :param batch_idx: The index of the current batch.
    #     :return: A tensor of losses between model predictions and targets.
    #     """
    #     loss, preds, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.train_loss(loss)
    #     self.train_acc(preds, targets)
    #     self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

    #     # return loss or backpropagation will fail
    #     return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    # def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    #     """Perform a single validation step on a batch of data from the validation set.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target
    #         labels.
    #     :param batch_idx: The index of the current batch.
    #     """
    #     loss, preds, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.val_loss(loss)
    #     self.val_acc(preds, targets)
    #     self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
    #     """Perform a single test step on a batch of data from the test set.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target
    #         labels.
    #     :param batch_idx: The index of the current batch.
    #     """
    #     loss, preds, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.test_loss(loss)
    #     self.test_acc(preds, targets)
    #     self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # _ = MNISTLitModule(None, None, None, None)
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "cifar10.yaml")
    _ = hydra.utils.instantiate(cfg)