# MedMAE: A Self-Supervised Backbone for Medical Imaging Tasks

This repository has the implementation for the paper [MedMAE: A Self-Supervised Backbone for Medical Imaging Tasks](https://arxiv.org/abs/2407.14784): by Anubhav Gupta, Islam Osman, Mohamed S. Shehata, and John W. Braun.

#### Summary
* [Introduction](#introduction)
* [Code](#code)
* [Visualization](#visualization)
* [Results](#results)
* [Pre-trained Weights](#pre-trained-weights)
* [Model & Data Usage](#model&data-usage)

## Introduction
This paper proposes a a large-scale unlabeled medical imaging dataset called MID that is collected from various sources, and a pre-trained backbone that is trained using [MAE](https://arxiv.org/abs/2111.06377) on the proposed dataset.
MID is collected from the following sources:
<p align="center">
    <img src="https://github.com/islamosmanubc/MedMAE/blob/main/figures/mid.png" width="700"/>
</p>
> Figure: A detailed overview of the various datasets collected to form MID the medical imaging dataset.


The link to download MID is [here](https://doi.org/10.20383/103.01017) (COMING SOON!).

We used MAE to pre-train a ViT-B backbone on the proposed dataset to allow the proposed model to gain useful knowledge of different types of medical image. Hence, the backbone can be used for any medical imaging task and acheive better performance than existing pre-trained models using ImageNet dataset.
<p align="center">
    <img src="https://github.com/islamosmanubc/MedMAE/blob/main/figures/mae.png" width="800"/>
</p>

> Figure: MedMAE architecture: The process is initiated by randomly
masking 75\% of the original image and inputting the remaining 25\% of visible patches into the encoder, which captures the latent representations and encodes the patches. Subsequently, the aim of the decoder is to reconstruct the complete image using the encoded and masked patches. The reconstruction loss helps to improve the reconstruction with each iteration.

## Code
We used the code in the github repository [MAE](https://github.com/facebookresearch/mae) and trained the model for 1000 epochs on the proposed dataset.

For linear probing on your own dataset: run the file 'main_linprobe_classification.py' set the argument 'data_path' with your dataset folder path and set 'nb_classes' to the number of classes in your dataset and set 'finetune' with the path to the pre-trained file on our dataset (check the Pre-trained weights section)

## Visualization
<p align="center">
    <img src="https://github.com/islamosmanubc/MedMAE/blob/main/figures/rec_mae.png" width="800"/>
</p>

> Figure: Image construction using pre-trained MAE with natural images.

<p align="center">
    <img src="https://github.com/islamosmanubc/MedMAE/blob/main/figures/rec_medmae.png" width="800"/>
</p>

> Figure: Image construction using our proposed MedMAE..



## Results
We report the results of our model in comparison with existing models on four different medical imaging tasks:
* Task 1: automating quality control for CT and MRI scanners (Private dataset)
  * Task 1.1: CT scans
  * Task 1.2: MRI scans
* Task 2: breast cancer prediction from CT images (Private dataset)
* Task 3: pneumonia detection from chest x-ray images (Public dataset: ChestX-ray14)
* Task 4: polyp segmentation in colonscopy sequences (Public dataset: CVC-ClinicDB)

|          Methods       |      Task 1.1    |     Task 1.2    |    Task 2   |    Task 3    |    Task 4    | 
| ---------------------- | ---------------- | --------------- | ----------- | ------------ | ------------ |
| `ResNet`               |      `0.756`     |      `0.716`    |   `0.799`   |   `0.664`    |   `0.579`    |
| `EfficientNet-S`       |      `0.713`     |      `0.661`    |   `0.751`   |   `0.629`    |   `0.531`    |
| `ConvNext-B`           |      `0.768`     |      `0.719`    |   `0.838`   |   `0.675`    |   `0.618`    |
| `ViT-B`                |      `0.782`     |      `0.727`    |   `0.840`   |   `0.678`    |   `0.635`    |
| `Swin-B`               |      `0.775`     |      `0.721`    |   `0.813`   |   `0.671`    |   `0.602`    |
| `MAE`                  |      `0.785`     |      `0.743`    |   `0.843`   |   `0.679`    |   `0.646`    |
| `MedMAE(ours)`         |      `0.902`     |      `0.856`    |   `0.932`   |   `0.701`    |   `0.714`    |


## Pre-trained Weights
The pre-trained weights are available on this link:
[pre-trained weights](https://drive.google.com/drive/folders/1ym3GdaI69NPhxX4l0ulr1oS0OWoBlhF-?usp=drive_link)

## Model & Data Usage
For using the pre-trained model or the proposed data you need to include the following citation:
```
@Article{islamosmanmedmae2024,
  author  = {Anubhav Gupta and Islam Osman and Mohamed S. Shehata and John W. Braun},
  journal = {arXiv:2407.14784},
  title   = {MedMAE: A Self-Supervised Backbone for Medical Imaging Tasks},
  year    = {2024},
}
```

The pre-trained weights on our medical dataset is 'pre_trained_medmae.pth'

To reproduce the reported results on chestX-ray14 and CVC-ClinicDB. First, load the base model 'pre_trained_medmae.pth'. Then, the weights for the target dataset 'chestx-medmae_finetune' or 'cvc-clinicdb_medmae'

