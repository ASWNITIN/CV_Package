import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class LowLightDataset(Dataset):

    def __init__(self, root_dir, mode='train', transform=None):
        super(LowLightDataset, self).__init__()

        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            self.input_dir = os.path.join(root_dir, 'Train', 'input')
            self.target_dir = os.path.join(root_dir, 'Train', 'target')
            self.input_images = sorted(os.listdir(self.input_dir))
            self.target_images = sorted(os.listdir(self.target_dir))
            assert len(self.input_images) == len(
                self.target_images
            ), "Number of input and target images should match for training."
        elif mode == 'test':
            self.input_dir = os.path.join(root_dir, 'Test', 'input')
            self.input_images = sorted(os.listdir(self.input_dir))
        else:
            raise ValueError('Mode should be "train" or "test"')

        # Default Transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        input_img = Image.open(input_path).convert('RGB')

        if self.mode == 'train':
            target_path = os.path.join(self.target_dir,
                                       self.target_images[idx])
            target_img = Image.open(target_path).convert('RGB')

            if self.transform:
                input_img = self.transform(input_img)
                target_img = self.transform(target_img)

            return input_img, target_img, self.input_images[idx]
        else:
            if self.transform:
                input_img = self.transform(input_img)

            return input_img, self.input_images[idx]


def get_dataloaders(root_dir,
                    batch_size=4,
                    num_workers=4,
                    train_transform=None,
                    test_transform=None):
    train_dataset = LowLightDataset(root_dir=root_dir,
                                    mode='train',
                                    transform=train_transform)
    test_dataset = LowLightDataset(root_dir=root_dir,
                                   mode='test',
                                   transform=test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    # Testing the dataset loader
    root = 'dataset'
    train_loader, test_loader = get_dataloaders(root_dir=root,
                                                batch_size=2,
                                                num_workers=0)

    for i, (inputs, targets, names) in enumerate(train_loader):
        print(f"Train batch {i + 1}:")
        print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
        print(f"Image names: {names}")
        if i == 1:
            break

    for i, (inputs, names) in enumerate(test_loader):
        print(f"Test batch {i + 1}:")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Image names: {names}")
        if i == 1:
            break
