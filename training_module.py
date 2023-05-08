import logging
import os
import sqlite3

import gdown
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from PIL import ImageFile
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10


class CollectionsDataset(Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 num_classes,
                 transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data.loc[idx, 'image_path'])
        image = Image.open(img_name).convert('RGB')
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[self.data.loc[idx, 'rating'] - 1] = 1

        if self.transform:
            image = self.transform(image)

        return {'image': image,
                'labels': label_tensor
                }


def train_model(model,
                data_loader,
                dataset_size,
                optimizer,
                scheduler,
                num_epochs):
    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        model.train()
        running_loss = 0.0
        # Iterate over data.
        for bi, d in enumerate(data_loader):
            inputs = d["image"]
            labels = d["labels"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_size
        scheduler.step()
        logging.info('Loss: {:.4f}'.format(epoch_loss))
    return model


def train_from_resnet101_with_dataset():
    model = get_model_definition()

    # define some re-usable stuff
    IMAGE_SIZE = 512
    BATCH_SIZE = 9
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # make some augmentations on training data
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    # use the collections dataset class we created earlier
    CSV_FILE_TRAIN = r"dataset\balanced\data.csv"
    ROOT_DIR_TRAIN = r"dataset\balanced"

    train_dataset = CollectionsDataset(CSV_FILE_TRAIN, ROOT_DIR_TRAIN, NUM_CLASSES, train_transform)

    # create the pytorch data loader
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True)
    # push model to device
    model = model.to(device)

    plist = [
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 5e-3}
    ]
    optimizer_ft = optim.Adam(plist, lr=0.001)
    lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model = train_model(model,
                        train_dataset_loader,
                        len(train_dataset),
                        optimizer_ft,
                        lr_sch,
                        num_epochs=5)
    return model


def train_model_from_ram_with_flagged(model, transforms, num_epochs=5, batch_size=9, lrs=[1e-5, 5e-3, 1e-3]):
    sql_to_csv()
    CSV_FILE_TRAIN = os.path.join("flagged", "data.csv")
    ROOT_DIR_TRAIN = ""
    train_dataset = CollectionsDataset(CSV_FILE_TRAIN, ROOT_DIR_TRAIN, NUM_CLASSES, transforms)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)

    plist = [
        {'params': model.layer4.parameters(), 'lr': lrs[0]},
        {'params': model.fc.parameters(), 'lr': lrs[1]}
    ]
    optimizer_ft = optim.Adam(plist, lr=lrs[2])
    lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model = train_model(model,
                        train_dataset_loader,
                        len(train_dataset),
                        optimizer_ft,
                        lr_sch,
                        num_epochs=num_epochs)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def train_model_from_my_pretrained(transforms, num_epochs=5, batch_size=9, lrs=[1e-5, 5e-3, 1e-3]):
    if not os.path.exists("model_pretrained.bin"):
        logging.info("Missing model, downloading")
        url = 'https://drive.google.com/uc?id=1uyIYjcPRg6TwIrLpa_p9bmCALtAax79F'
        output = 'model_pretrained.bin'
        gdown.download(url, output, quiet=False)
        logging.info("Model downloaded")
    model = get_model_definition()
    model.load_state_dict(torch.load("model_pretrained.bin"))
    model = model.to(device)
    sql_to_csv()
    CSV_FILE_TRAIN = os.path.join("flagged", "data.csv")
    ROOT_DIR_TRAIN = ""
    train_dataset = CollectionsDataset(CSV_FILE_TRAIN, ROOT_DIR_TRAIN, NUM_CLASSES, transforms)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)

    plist = [
        {'params': model.layer4.parameters(), 'lr': lrs[0]},
        {'params': model.fc.parameters(), 'lr': lrs[1]}
    ]
    optimizer_ft = optim.Adam(plist, lr=lrs[2])
    lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model = train_model(model,
                        train_dataset_loader,
                        len(train_dataset),
                        optimizer_ft,
                        lr_sch,
                        num_epochs=num_epochs)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def get_model_definition():
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=10),
    )
    return model


def sql_to_csv():
    conn = sqlite3.connect(os.path.join("flagged", "data.db"), isolation_level=None,
                           detect_types=sqlite3.PARSE_COLNAMES)
    db_df = pd.read_sql_query("SELECT * FROM images", conn)
    db_df.to_csv(os.path.join("flagged", "data.csv"), index=False)
