# Diffusion_

Start main for model to learn and create images (set up config/params beforehand in json)

Use create_mnist_images afterwards to create folders with training/tests images 

Use 'python FID\fid.py --p1  ...\DiffusionModel\test_img\test_img\ --p2 ...\DiffusionModel\OutputSamples\steps_{number of steps} -b {batches used in config}' to get value of FID

wandb.ai: https://wandb.ai/ivan_ch/DiffusionModelMNIST?workspace=user-ivan_ch


versions:
* Version 0 { batch: 128, steps: 1000, epoch: 30, learning_rate: 2e-4}