# Emotion Analysis

This project includes a model training tool to train models to anlayse emotions from face pictures.
The dataset used for training is FER2013.

## Quick Start

1.Install dependencies

```bash
python -m pip install -r requirements.txt
```

2.Train a single model:

```bash
python -m src.training.train
```

Options:

```bash
  -h, --help            show this help message and exit
  --seed SEED           Random seed for reproducibility
  --dataset {fer}
                        Dataset name (default: fer)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs to wait for improvement before early stopping
  --output_dir OUTPUT_DIR
                        Directory to save checkpoints and logs
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --epochs EPOCHS       Number of epochs to train (default: 100)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.001)
  --model {vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,efficientnet,mobilenet,mobilefacenet}
                        Model architecture to use (default: vgg16)
  --pretrained          Use pretrained weights for the model
  --freezed             Freeze the convolutional layers of the model
  --num_outputs NUM_OUTPUTS
                        Number of outputs for the regression head (default: 3)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the regression head (default: 0.5)
  --optimizer {adam,sgd,adamw}
                        Optimizer to use (default: adam)
  --cuda                Use CUDA for training if available
  --grad_clip GRAD_CLIP
                        Gradient clipping value (default: 0.0, no clipping)
  --data_augmentation_param DATA_AUGMENTATION_PARAM
                        Data augmentation parameter for rotation (degrees), scaling (percent), and shifting (percent) (default: 5)
  --resume              Resume training from the last checkpoint if available
  --no_checkpoint       Disable checkpoint saving
  --no_model_save       Disable model saving
  --lr_factor LR_FACTOR
                        Factor by which to reduce learning rate (default: 0.1)
  --lr_patience LR_PATIENCE
                        Number of epochs to wait for improvement before reducing learning rate (default: 10)
  --lr_threshold LR_THRESHOLD
                        Minimum change in loss to qualify as improvement (default: 1e-4)
  --lr_threshold_mode LR_THRESHOLD_MODE
                        Mode to use for determining if loss has improved (default: rel)
  --lr_cooldown LR_COOLDOWN
                        Number of epochs to wait before resuming normal operation after reducing learning rate (default: 0)
  --lr_min LR_MIN       Minimum learning rate (default: 0.0)
```

## Repository Layout

- `data/`: untracked, store the dataset csvs generated with /src/data/prepare_datasets.py here
- `models/`: model architectures for age estimation
- `output/`: training logs and model weights
- `src/data/`: data preparation and preprocessing
- `src/training/`: training scripts
- `src/transforms/`: transforms methods for data loading and data augmentation
- `src/utils/`: shared dataset and utility modules
