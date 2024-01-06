# DL-521153S-F2023 - Summary

In the pursuit of optimizing model performance for image classification tasks, our deep learning project underwent a comprehensive exploration encompassing pre-training, fine-tuning, and critical observations. This report highlights key findings and outcomes from each phase, underscoring the nuances of model behavior across different architectures and datasets.

## Pre-training strategies

### Raw model training

In the preliminary phase, rigorous experimentation with learning rates ranging from 1e-8 to 1e-1 was conducted. Notably, a learning rate of 1e-2 emerged as optimal, resulting in a validation accuracy of 50%. Despite the investment of substantial time, neither of the models surpassed the 50% accuracy threshold, hinting at inherent complexities and challenges.

### Pre-training with Pretrained weights

#### resnet18

The utilization of pretrained weights exhibited promising progress, with the initial 7 epochs showcasing positive training trends. The model stabilized at a commendable validation accuracy of 76.98%.

#### resnet50

Training on the mini_imagenet dataset using resnet50 yielded impressive results. After just 2 epochs, the model achieved a notable validation accuracy of 86.30%.

#### efficientnet_b0

performance displayed variability during training on the mini_imagenet dataset. Starting with a modest 33.96% accuracy after the first epoch, it steadily improved to 75.86% by the fourth epoch.

## Fine-Tuning Analysis

### efficientnet_b0

Following fine-tuning on the EuroSAT dataset, efficientnet_b0 demonstrated an average accuracy of 17.11%, with the highest accuracy reaching 70.67%. Notably, the model struggled with generalization, potentially attributed to a pronounced domain shift between the mini_imagenet and EuroSAT datasets.

### resnet18

Fine-tuning resnet18 on EuroSAT produced compelling results, boasting an average accuracy of 71.20% and a peak accuracy of 96.00%. This underscores resnet18's adeptness at adapting to novel datasets and domains.

### resnet50

Fine-tuning resnet50 on EuroSAT resulted in an average accuracy of 64.57%, with the highest accuracy also peaking at 96.00%. While showcasing commendable adaptability, the average accuracy marginally lagged behind resnet18, possibly due to the model's increased complexity.


# Outputs

### resnet18

```bash
python main.py --task train --model resnet18 --pretrained
```

```python
DEBUG: Using 'cuda' for inference
DEBUG: Using resnet18 model
DEBUG: Enabling pretrained weights for training
Start pretraining and eval using model resnet18...
Warning: pretrain data already exist, data will be replaced!
epoch 1/25 => train loss: 2.008707, val loss: 1.201550, val accuracy: 70.68%
epoch 2/25 => train loss: 0.921940, val loss: 1.067962, val accuracy: 74.43%
epoch 3/25 => train loss: 0.628170, val loss: 0.942850, val accuracy: 75.70%
epoch 4/25 => train loss: 0.431880, val loss: 0.823947, val accuracy: 76.35%
epoch 5/25 => train loss: 0.285835, val loss: 0.807467, val accuracy: 76.82%
epoch 6/25 => train loss: 0.177270, val loss: 0.824640, val accuracy: 76.98%
epoch 7/25 => train loss: 0.163811, val loss: 0.802346, val accuracy: 76.98%
...
```

### resnet50

```bash
python main.py --task train --model resnet50 --pretrained
```

```python
DEBUG: Using 'cuda' for inference
DEBUG: Using resnet50 model
DEBUG: Enabling pretrained weights for training
Start pretraining and eval using model resnet50...
Warning: pretrain data already exist, data will be replaced!
epoch 1/25 => train loss: 1.566956, val loss: 0.799155, val accuracy: 84.61%
epoch 2/25 => train loss: 0.506727, val loss: 0.615288, val accuracy: 86.30%
...
```

### efficientnet_b0

```bash
python main.py --task train --model efficientnet_b0 --pretrained
```

```python
DEBUG: Using 'cuda' for inference
DEBUG: Using efficientnet_b0 model
DEBUG: Enabling pretrained weights for training
Start pretraining and eval using model efficientnet_b0...
Warning: pretrain data already exist, data will be replaced!
epoch 1/25 => train loss: 6.710531, val loss: 5.070756, val accuracy: 33.96%
epoch 2/25 => train loss: 4.093973, val loss: 2.748469, val accuracy: 65.16%
epoch 3/25 => train loss: 3.127901, val loss: 2.060424, val accuracy: 73.15%
epoch 4/25 => train loss: 2.738979, val loss: 1.799939, val accuracy: 75.86%
...
```

# tuning

```bash
python main.py --task tune --model efficientnet_b0
```

```python
DEBUG: Using 'cuda' for inference
DEBUG: Using efficientnet_b0 model
Start tunning pretrained data for model efficientnet_b0...
Run 256: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [01:38<00:00,  2.61Run/s, accuracy=34.67%]
fine-tune model average accuracy: 17.11%, best accuracy: 70.67%
```

```bash
python main.py --task tune --model resnet18
```

```python
DEBUG: Using 'cuda' for inference
DEBUG: Using resnet18 model
Start tunning pretrained data for model resnet18...
Run 256: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [01:32<00:00,  2.77Run/s, accuracy=89.33%]
fine-tune model average accuracy: 71.20%, best accuracy: 96.00%
```

```bash
python main.py --task tune --model resnet50
```

```python
DEBUG: Using 'cuda' for inference
DEBUG: Using resnet50 model
Start tunning pretrained data for model resnet50...
Run 256: 100%|████████████████████████████████████████████████████████████████████████████████████████
fine-tune model average accuracy: 64.57%, best accuracy: 96.00%
```
