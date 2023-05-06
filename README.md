# Diffusion_

Start main for model to learn and create images (set up config/params beforehand in json)

Use create_mnist_images afterwards to create folders with training/tests images 

Use 'python FID\fid.py --p1  ...\DiffusionModel\test_img\test_img\ --p2 ...\DiffusionModel\OutputSamples\steps_{number of steps} -b {batches used in config}' to get value of FID (absolute paths)

wandb.ai: https://wandb.ai/ivan_ch/DiffusionModelMNIST?workspace=user-ivan_ch
https://wandb.ai/ivan_ch/DiffusionModelFoldImages32?workspace=user-ivan_ch

test_img, train_img: raw samples of MNIST; FID = FID =  1.1078643047784453 (between folders) 
train, val: raw samples of FoldImages32x32; FID = 5.884192542744302 (between folders) 

MNIST
versions:
* Version 0 { batch: 128, steps: 1000, epoch: 30, learning_rate: 2e-4}; FID =  19.99149611355972 (between generated and test_img)


FoldImages32:
* Version 0 = {
  "diffusion_steps": 1000,
  "max_epoch": 10,
  "batch_size": 64,
  "picture_size": [32, 32]
},  learning_rate: 2e-4; FID = 180.53354925784905

* Version 4 = {
  "diffusion_steps": 1000,
  "max_epoch": 10,
  "batch_size": 128,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 162.79946719161373

* Version 5 = {
  "diffusion_steps": 1000,
  "max_epoch": 10,
  "batch_size": 64,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 162.79946719161373


* Version 6 = {
  "diffusion_steps": 1000,
  "max_epoch": 25,
  "batch_size": 64,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 113.22502025710958


* Version 7 = {
  "diffusion_steps": 1000,
  "max_epoch": 50,
  "batch_size": 64,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 73.27725868675091
Transform = 73.74181602016901


* Version 8 = {
  "diffusion_steps": 1000,
  "max_epoch": 100,
  "batch_size": 64,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 84.79603973304424

* Version 8 (new) = {
  "diffusion_steps": 1500,
  "max_epoch": 50,
  "batch_size": 64,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 73.76131269149471 (generating 2500 images, took 41m)

* Version 9 = {
  "diffusion_steps": 1000,
  "max_epoch": 50,
  "batch_size": 128,
  "picture_size": [32, 32],
  "learning_rate": 2e-4
}, FID = 260.34303517745775