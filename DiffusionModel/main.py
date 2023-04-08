from Data.data import DiffSet
from Model.model import DiffusionModel

import torch
import glob
import json
import os
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    params = json.load(open('params.json'))

    diffusion_steps = params['diffusion_steps']
    dataset_choice = "MNIST"
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']


    # Loading parameters
    load_model = params['load_model']
    load_version_num =  params['load_version_num']

    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    if load_model:
        pass_version = load_version_num
        last_checkpoint = glob.glob(
            f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
        )[-1]

    # Create datasets and data loaders
    train_dataset = DiffSet(True, dataset_choice)
    val_dataset = DiffSet(False, dataset_choice)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create model and trainer
    if load_model:
        model = DiffusionModel.load_from_checkpoint(last_checkpoint, in_size=train_dataset.size * train_dataset.size,
                                                    t_range=diffusion_steps, img_depth=train_dataset.depth)
    else:
        model = DiffusionModel(train_dataset.size * train_dataset.size, diffusion_steps, train_dataset.depth)

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        "/lightning_logs/",
        name=dataset_choice,
        version=pass_version,
    )

    trainer = pl.Trainer(
        max_epochs=max_epoch,
        log_every_n_steps=8,
        accelerator="gpu",
        devices="auto",
        logger=tb_logger
    )

    model.cuda()

    # Train model
    if load_model:
      model.eval()
    else:
      trainer.fit(model, train_loader, val_loader)

    create_sample_batch_size = params['create_sample_batch_size']

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn((create_sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size)).to(torch.device('cuda'))
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

    path = os.path.abspath(os.getcwd())

    if not os.path.exists(path + f'/OutputSamples/steps_{diffusion_steps}'):
        os.makedirs(path + f'/OutputSamples/steps_{diffusion_steps}')

    from datetime import datetime
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for i in range(gen_samples.shape[0]):
        for j in range(gen_samples.shape[1]):
            new_image = Image.fromarray(gen_samples[i][j].cpu().numpy())
            new_image.save(path + f'/OutputSamples/steps_{diffusion_steps}/{time}_' + str(i) + "_" + str(j) + ".png")
