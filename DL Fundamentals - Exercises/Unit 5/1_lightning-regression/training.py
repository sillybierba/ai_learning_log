# Unit 5.5. Organizing Your Data Loaders with Data Modules

import lightning as L
import torch
from shared_utilities import LightningModel, AmesHousingDataModule, PyTorchMLP
from watermark import watermark

if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    torch.manual_seed(123)

    dm = AmesHousingDataModule()

    pytorch_model = PyTorchMLP(num_features=3)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cpu", devices="auto", deterministic=True
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_mse = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_mse"]
    val_mse = trainer.validate(datamodule=dm)[0]["val_mse"]
    test_mse = trainer.test(datamodule=dm)[0]["test_mse"]
    print(
        f"Train MSE {train_mse:.2f}%"
        f" | Val MSE {val_mse:.2f}%"
        f" | Test MSE {test_mse:.2f}%"
    )