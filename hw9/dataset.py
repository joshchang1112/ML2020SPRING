from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images
