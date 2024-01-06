import os
import multiprocessing

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset

import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, resnet18, efficientnet_b0
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomVerticalFlip, ToTensor, Normalize

import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from models.ResNet import ResNet18, ResNet50
from models.EfficientNet import EfficientNetB0
from config import PRETRAIN_DIRECTORY, DATA_DIRECTORY

class Trainer:
    def __init__(self, model, batch_size, image_size, num_epochs, learning_rate, pretrained):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.pretrain_path = os.path.join(PRETRAIN_DIRECTORY, "%s_miniimagenet.pth" % model)

        self.num_workers = multiprocessing.cpu_count()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            # transforms.RandomResizedCrop(84, scale=(0.7, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomRotation(10),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ToTensor(),
        ])

        full_dataset = ImageFolder(root=os.path.join(DATA_DIRECTORY, 'train'), transform=self.transform)

        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=42)
        indices = list(stratified_splitter.split(full_dataset, full_dataset.targets))

        train_indices, val_test_indices = indices[0]
        val_indices, test_indices = random_split(val_test_indices, [val_size, test_size])

        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)
 
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"DEBUG: Using '{self.device}' for inference")

        self.model_class = None

        if model == "resnet18":
            self.model_class = ResNet18
        if model == "resnet50":
            self.model_class = ResNet50
        if model == "efficientnet_b0":
            self.model_class = EfficientNetB0

        if not self.model_class:
            print("model %s is not supported at the moment!" % model)
            exit(1)

        print(f"DEBUG: Using {model} model")
        
        if pretrained:
            print("DEBUG: Enabling pretrained weights for training")

        self.num_classes = len(full_dataset.classes)
        self.model = self.model_class(num_classes=self.num_classes, pretrained=pretrained)
        self.model.to(self.device)

        # self.optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=0.0005)
        self.criterion = nn.CrossEntropyLoss()

        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.scaler = GradScaler()


    def pretrain_and_eval(self, num_epochs=25):
        best_val_loss = float('inf')
        best_epoch = 0

        # if pretrain data exist warn that we are going to replace it
        if os.path.isfile(self.pretrain_path):
            print("Warning: pretrain data already exist, data will be replaced!")


        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0

            with tqdm(self.train_loader, unit="batch", leave=False, desc=f"epoch {epoch + 1}/{num_epochs}", bar_format="{desc} =>{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]") as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with autocast():
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    total_train_loss += loss.item()
                    num_batches += 1


            avg_train_loss = total_train_loss / num_batches

            # val
            self.model.eval()
            val_predictions, val_labels = [], []

            with torch.no_grad():
                for val_inputs, val_labels_batch in self.val_loader:
                    val_inputs, val_labels_batch = val_inputs.to(self.device), val_labels_batch.to(self.device)
                    val_outputs = self.model(val_inputs)
                    _, val_preds = torch.max(val_outputs, 1)
                    val_predictions.extend(val_preds.cpu().numpy())
                    val_labels.extend(val_labels_batch.cpu().numpy())

            val_loss = self.criterion(val_outputs, val_labels_batch)
            val_accuracy = accuracy_score(val_labels, val_predictions)

            print(f"epoch {epoch + 1}/{num_epochs} => "
                  f"train loss: {avg_train_loss:4f}, "
                  f"val loss: {val_loss:4f}, "
                  f"val accuracy: {val_accuracy:.2%}")

            self.scheduler.step()

            # save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.pretrain_path)

        # after training call the test method
        self.test()


    def test(self):
        # load pretrained data
        self.model.load_state_dict(self.get_pretrained_data())

        # testing
        self.model.eval()

        test_predictions, test_labels = [], []
        with torch.no_grad():
            for test_inputs, test_labels_batch in self.test_loader:
                test_inputs, test_labels_batch = test_inputs.to(self.device), test_labels_batch.to(self.device)
                test_outputs = self.model(test_inputs)
                _, test_preds = torch.max(test_outputs, 1)
                test_predictions.extend(test_preds.cpu().numpy())
                test_labels.extend(test_labels_batch.cpu().numpy())

        test_accuracy = accuracy_score(test_labels, test_predictions)
        print(f"Test Accuracy: {test_accuracy:.2%}")
        return test_accuracy


    def get_pretrained_data(self):
        if not os.path.isfile(self.pretrain_path):
            print(f"{self.pretrain_path} does not exist! please run with --train first :)")
            print("exiting...")
            exit(1)

        return torch.load(self.pretrain_path)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
               nn.init.constant_(m.bias, 0)
