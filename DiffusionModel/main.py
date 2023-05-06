from Data.data import DiffSet, DiffDataModule
from Model.model import DiffusionModel

import torch
import glob
import json
import os
import shutil
import pytorch_lightning as pl
from PIL import ImageColor
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from torch import save


def get_parent_folder() -> str:
    """Возвращает абсолютный путь к директории, содержащей этот файл

    Returns:
        str: абсолютный путь к директории, содержащей этот файл
    """
    return Path(__file__).parent

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    absolute_path = str(get_parent_folder())
    params = json.load(open(absolute_path + '/params.json'))

    diffusion_steps = params['diffusion_steps']
    dataset_choice = params['name_dataset']
    path2data = params['path2data']
    picture_size = params['picture_size']
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    additional = params['additional']

    # Loading parameters
    load_model = params['load_model']
    load_version_num = params['load_version_num']

    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    if load_model:
        pass_version = load_version_num

    # Create datasets and data loaders
    # train_dataset = DiffSet(True, dataset_choice)
    # val_dataset = DiffSet(False, dataset_choice)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    # def __init__(self,
    #          absolute_path: str,
    #          name_dataset: str,
    #          path2data: str,
    #          batch_size: int = 32,
    #          picture_size: [int, int] = [256, 256],
    #          shuffle: bool = True,
    #          transform: transforms.Compose = None):
    DDM = DiffDataModule(absolute_path, dataset_choice, path2data, batch_size, picture_size)

    # Create model and trainer
    model = DiffusionModel(learning_rate, picture_size[0] * picture_size[1], diffusion_steps, DDM.train.depth)
    if load_model:
      model.load_state_dict(torch.load(absolute_path + f'/Model/SavedWeights/{dataset_choice}/version_{load_version_num}/weights_biases.pth'))

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        absolute_path + "/lightning_logs/",
        name=dataset_choice,
        version=pass_version,
    )
    
    wandb_logger = WandbLogger(project='DiffusionModel' + dataset_choice, job_type='train')

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        log_every_n_steps=5,
        accelerator="gpu",
        devices="auto",
        logger=[tb_logger, wandb_logger],
        enable_checkpointing=False
    )

    from datetime import datetime
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Train model
    if load_model:
      # model.eval()
      print()
    else:
      trainer.fit(model, DDM)    
      current_path = absolute_path + f'/Model/SavedWeights/{dataset_choice}/version_{load_version_num}'
      if not os.path.exists(current_path):
        os.makedirs(current_path)
      save(model.state_dict(), current_path + '/weights_biases.pth')
      shutil.copyfile(absolute_path + '/params.json', current_path + '/params.json')


    model.eval()
    model.cuda()
    create_sample_batch_size = params['create_sample_batch_size']

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((create_sample_batch_size, DDM.train.depth, picture_size[0], picture_size[1])).to(torch.device('cuda'))
    sample_steps = torch.arange(model.t_range - 1, 0, -1).to(torch.device('cuda'))

    for t in sample_steps:
        if t % 100 == 0:
          print(t)
        x = model.denoise_sample(x, t)
        if t == 1:
            gen_samples.append(x)

    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    gen_samples = (gen_samples * 255).type(torch.uint8)

    if not os.path.exists(absolute_path + f'/OutputSamples/{dataset_choice}/version_{load_version_num}_steps_{diffusion_steps}_{additional}'):
        os.makedirs(absolute_path + f'/OutputSamples/{dataset_choice}/version_{load_version_num}_steps_{diffusion_steps}_{additional}')

    for i in range(gen_samples.shape[0]):
        for j in range(gen_samples.shape[1]):
            # print(gen_samples[i][j].cpu().numpy())
            new_image = Image.fromarray(gen_samples[i][j].cpu().numpy())
            new_image.save(absolute_path + f'/OutputSamples/{dataset_choice}/version_{load_version_num}_steps_{diffusion_steps}_{additional}/{time}_' + str(i) + "_" + str(j) + ".png")
            # new_image = DDM.train.inverse_transform(gen_samples[i][j].cpu().numpy())
            # new_image.save(absolute_path + f'/OutputSamples/{dataset_choice}/version_{load_version_num}_steps_{diffusion_steps}_{additional}/{time}_' + str(i) + "_" + str(j) + ".png")
