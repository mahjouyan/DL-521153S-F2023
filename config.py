import os

DATASET = dict([
        ("train.tar", "https://docs.google.com/uc?export=download&id=107FTosYIeBn5QbynR46YG91nHcJ70whs"),
        # ("test.tar", "https://docs.google.com/uc?export=download&id=1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v"),
        # ("val.tar", "https://docs.google.com/uc?export=download&id=1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl"),
        ("EuroSAT_RGB.zip", "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1")
        ])

DATA_DIRECTORY = os.path.join(os.getcwd(), 'data')
PRETRAIN_DIRECTORY = os.path.join(os.getcwd(), 'pretrain')
