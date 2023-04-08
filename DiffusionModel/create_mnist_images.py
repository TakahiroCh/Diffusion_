from torchvision import datasets, transforms
import os

if __name__ == '__main__':
    train_data = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=False)
    test_data = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=False)
    for i in [train_data, test_data]:
        if i == train_data:
            count = 0
            for i, (image, label) in enumerate(train_data):
                img = transforms.ToPILImage()(image)
                if not os.path.exists(f'./train_img/'):
                    os.makedirs(f'./train_img/')
                else:
                    pass
                img.save(f'./train_img/{count}.png')
                count += 1
        else:
            count = 0
            for i, (image, label) in enumerate(test_data):
                img = transforms.ToPILImage()(image)
                if not os.path.exists(f'./test_img/'):
                    os.makedirs(f'./test_img/')
                else:
                    pass
                img.save(f'./test_img/{count}.png')
                count += 1
