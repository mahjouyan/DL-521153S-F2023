import os
import random

from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score

from trainer import Trainer
from config import DATA_DIRECTORY


class Tunner:
    def __init__(self, trainer, dataset, batch_size, learning_rate, num_tune_runs):
        if not trainer:
            print("Trainer/Pretrained data model is required!")
            exit(1)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_runs = num_tune_runs

        self.num_categories = 5
        self.num_images_per_category = 20
        self.num_tun_per_category = 5

        self.trainer = trainer

        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        if dataset == "eurosat":
            self.dataset = ImageFolder(root=os.path.join(DATA_DIRECTORY, 'EuroSAT_RGB'), transform=transform)


        if not self.dataset:
            print("dataset %s is not supported at the moment!" % dataset)
            exit(1)

    def get_sample_dataset(self):
        # ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
        all_categories = self.dataset.classes

        if self.num_categories > len(all_categories):
           print("num_categories is bigger than the number of all categories %s", str(len(all_categories)))
           exit(1)

        # select random categories
        selected_categories = random.sample(all_categories, self.num_categories)

        tun_data = []
        test_data = []

        for category in selected_categories:
            category_indices = [i for i, (_, label) in enumerate(self.dataset.samples) if self.dataset.classes[label] == category]
            category_items = random.sample(category_indices, self.num_images_per_category)

            # get tunning data for each category items
            tun_dataset_category = random.sample(category_items, self.num_tun_per_category)
            tun_data.extend(tun_dataset_category)

            # rest we use as test data
            test_data.extend([item for item in category_items if item not in tun_dataset_category])


        tun_dataset = Subset(self.dataset, indices=tun_data)
        test_dataset = Subset(self.dataset, indices=test_data)

        return tun_dataset, test_dataset, all_categories


    def start(self):
        # track of average accuracy
        total_accuracy = 0.0
        best_tune_accuracy = 0.0

        model = self.trainer.model_class(num_classes=self.trainer.num_classes, pretrained=False)
        model = model.to(self.device)

        # load pretrained state to the model
        model.load_state_dict(self.trainer.get_pretrained_data())

        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        criterion = nn.CrossEntropyLoss()

        # fine tune multiple times and get the result
        with trange(self.num_runs, unit="Run") as pbar:
            for run in pbar:
                pbar.set_description(f"Run {run + 1}")

                # get random sample data
                tun_dataset, test_dataset, all_categories = self.get_sample_dataset()

                tune_train_loader = DataLoader(tun_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.trainer.num_workers)
                tune_test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.trainer.num_workers)

                model.train()

                for inputs, labels in tune_train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()

                test_predictions, test_labels = [], []
                with torch.no_grad():
                    for test_inputs, test_labels_batch in tune_test_loader:
                        test_inputs, test_labels_batch = test_inputs.to(self.device), test_labels_batch.to(self.device)
                        test_outputs = model(test_inputs)
                        _, test_preds = torch.max(test_outputs, 1)
                        test_predictions.extend(test_preds.cpu().numpy())
                        test_labels.extend(test_labels_batch.cpu().numpy())

                accuracy = accuracy_score(test_labels, test_predictions)

                if accuracy > best_tune_accuracy:
                    best_tune_accuracy = accuracy

                pbar.set_postfix(accuracy=f"{accuracy:.2%}")

                total_accuracy += accuracy

        average_accuracy  = total_accuracy / self.num_runs
        print(f"fine tune model average accuracy: {average_accuracy:.2%}, best accuracy: {best_tune_accuracy:.2%}")

        pass

