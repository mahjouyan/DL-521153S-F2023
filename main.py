import os
import argparse
from pathlib import Path

from trainer import Trainer
from tunner import Tunner

from utils import download_dataset, check_dataset_files
from config import DATA_DIRECTORY


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
            '--tasks',
            action='store',
            dest='tasks',
            type=str,
            nargs='*',
            default=[],
            help="tasks to run including train, test, tune"
            )
    argParser.add_argument(
            "--download",
            action="store_true",
            help="download dataset")
    argParser.add_argument(
            "--model",
            action="store",
            choices=['resnet18', 'resnet50', 'efficientnet_b0'],
            default="resnet50",
            help="model to use on training/eval/test")
    argParser.add_argument(
            "--pretrained",
            action="store_true",
            help="load model for training with pretrained weight")
    argParser.add_argument(
            "--tune_dataset",
            action="store",
            choices=["eurosat"],
            default="eurosat",
            help="dataset for fine tuning")
    argParser.add_argument(
            "--batch_size",
            action="store",
            type=int,
            default=64,
            help="batch size")
    argParser.add_argument(
            "--image_size",
            action="store",
            type=int,
            default=84,
            help="batch size")
    argParser.add_argument(
            "--num_epochs",
            action="store",
            type=int,
            default=25,
            help="num of training epochs")
    argParser.add_argument(
            "--learning_rate",
            action="store",
            type=int,
            default=1e-3,
            help="learning rate")
    argParser.add_argument(
            "--num_tune_runs",
            action="store",
            type=int,
            default=256,
            help="num of runnig tune")

    args = argParser.parse_args()

    # check if data directory exist or not
    if not os.path.exists(DATA_DIRECTORY):
        print("Data directory does not exist, creating...")
        os.mkdir(DATA_DIRECTORY)


    # check if required models are already been downloaded
    missing_datasets = check_dataset_files()

    # if any datasetr is missing we need to download them
    if missing_datasets:
        if not args.download:
            print(
                    "Required dataset's (%s) not found, use 'download' argument to setup dataset's" % list(missing_datasets.keys()))
            print("Exiting...")
            exit(1)

        for file_name, url in missing_datasets.items():
            download_dataset(url, file_name)

        print("All dataset has been downloaded successfully!")

    # get trainer instance
    trainer = Trainer(model=args.model, batch_size=args.batch_size, image_size=args.image_size, num_epochs=args.num_epochs, learning_rate=args.learning_rate, pretrained=args.pretrained)
    tunner = Tunner(trainer=trainer, dataset=args.tune_dataset, learning_rate=args.learning_rate, batch_size=args.batch_size, num_tune_runs=args.num_tune_runs)

    if 'train' in args.tasks:
        print(f"Start pretraining and eval using model {args.model}...")
        trainer.pretrain_and_eval()


    if 'test' in args.tasks:
        print(f"Start testing pretrained data for model {args.model}...")
        trainer.test()


    if 'tune' in args.tasks:
        print(f"Start tunning pretrained data for model {args.model}...")
        tunner.start()



