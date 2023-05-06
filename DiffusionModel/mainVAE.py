from Data.data import DiffSet, DiffDataModule
from Model.model import DiffusionModel
from Model.VAE import VAE
from torchvision.utils import save_image

import torch
import glob
import json
import os
import shutil
import numpy
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
    params = json.load(open(absolute_path + '/paramsVAE.json'))

    dataset_choice = params['name_dataset']
    path2data = params['path2data']
    picture_size = params['picture_size']
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    additional = params['additional']
    hidden_size = params['hidden_size']
    vae_alpha = params['vae_alpha']

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
    #     def __init__(self,
                    #  picture_size: [int, int],
                    #  channels: int,
                    #  hidden_size: int,
                    #  alpha: int,
                    #  lr: float,
                    #  save_images: Optional[bool] = None,
                    #  save_path: Optional[str] = None):
    DDM = DiffDataModule(absolute_path, dataset_choice, path2data, batch_size, picture_size)

    path2images = absolute_path + f'/OutputSamplesVAE/{dataset_choice}/version_{load_version_num}_{additional}'
    # Create model and trainer
    model = VAE(picture_size, DDM.train.depth, hidden_size, vae_alpha, learning_rate, True, path2images)
    if load_model:
      model.load_state_dict(torch.load(absolute_path + f'/Model/SavedWeightsVAE/{dataset_choice}/version_{load_version_num}/weights_biases.pth'))

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        absolute_path + "/lightning_logs/",
        name=dataset_choice,
        version=pass_version,
    )
    
    wandb_logger = WandbLogger(project='VAEModel' + dataset_choice, job_type='train')

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
      current_path = absolute_path + f'/Model/SavedWeightsVAE/{dataset_choice}/version_{load_version_num}'
      if not os.path.exists(current_path):
        os.makedirs(current_path)
      save(model.state_dict(), current_path + '/weights_biases.pth')
      shutil.copyfile(absolute_path + '/paramsVAE.json', current_path + '/paramsVAE.json')


    model.eval()
    model.cuda()
    create_sample_batch_size = params['create_sample_batch_size']

    # Generate samples from denoising process
    gen_samples = []

    # for i in range(create_sample_batch_size):
    #   x = torch.randn((create_sample_batch_size, DDM.train.depth, picture_size[0], picture_size[1])).to(torch.device('cuda'))
    #   _, _, _gen_samples = model(x)
    #   gen_samples.append(_gen_samples)

    DDM.setup("test")
    step = round(numpy.floor(len(DDM.test) / batch_size))
    for i in range(step):
      x = [DDM.test.get_item(j) for j in range(batch_size)]
      _, _, _gen_samples = model(torch.stack(x).to(torch.device('cuda')))
      gen_samples.append(_gen_samples)

    if not os.path.exists(absolute_path + f'/OutputSamplesVAE/{dataset_choice}/version_{load_version_num}_{additional}'):
        os.makedirs(absolute_path + f'/OutputSamplesVAE/{dataset_choice}/version_{load_version_num}_{additional}')

    for k in range(len(gen_samples)):
      for i in range(gen_samples[k].shape[0]):
          for j in range(gen_samples[k].shape[1]):
              output_sample = gen_samples[k][i][j].reshape(-1, DDM.train.depth, picture_size[0], picture_size[1])
              # output_sample = self.scale_image(output_sample)
              save_image(
                  output_sample,
                  absolute_path + f'/OutputSamplesVAE/{dataset_choice}/version_{load_version_num}_{additional}/from_test_generation_{time}_' + str(k) + "_" + str(i) + "_" + str(j) + ".png",
              )

          # new_image = Image.fromarray(gen_samples[i][j].detach().cpu().numpy())
          # new_image.save(absolute_path + f'/OutputSamplesVAE/{dataset_choice}/version_{load_version_num}_{additional}/{time}_' + str(i) + "_" + str(j) + ".png")
          
          # new_image = DDM.train.inverse_transform(gen_samples[i][j].cpu().numpy())
          # new_image.save(absolute_path + f'/OutputSamples/{dataset_choice}/version_{load_version_num}_steps_{diffusion_steps}_{additional}/{time}_' + str(i) + "_" + str(j) + ".png")
