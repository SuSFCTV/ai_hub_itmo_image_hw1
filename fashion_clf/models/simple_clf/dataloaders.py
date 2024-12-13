from torch.utils.data import DataLoader

from fashion_clf.models.simple_clf.fashion_dataset import FashionDataset
import torchvision.transforms as transforms


def get_dataloaders(train_csv, test_csv):
    train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = DataLoader(train_set, batch_size=100)
    test_loader = DataLoader(test_set, batch_size=100)

    return train_loader, test_loader