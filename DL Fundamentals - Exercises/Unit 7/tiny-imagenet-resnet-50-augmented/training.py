import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torch
import torchvision
from local_utilities import LightningModel, TinyImageNetDataModule, plot_csv_logger

def get_torchvision_entrypoints():
    entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    for e in entrypoints:
        if "resnet" in e:
            print(e)


if __name__ == "__main__":
    dm = TinyImageNetDataModule(height_width=(224, 224), batch_size=64, num_workers=4)
   
    pytorch_model = torch.hub.load('pytorch/vision', 'resnet50', weights=None)
    L.pytorch.seed_everything(123)
    print(pytorch_model)
    # exit()

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.1)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        logger=CSVLogger(save_dir="logs/", name="my-model"),
        deterministic=True,
    )

    trainer.fit(model=lightning_model, datamodule=dm)
    torch.save(pytorch_model.state_dict(), "tiny-imagenet-resnet50-augmented.pt")
