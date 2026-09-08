# Emotion Analysis

This project includes a model training tool to train models to anlayse emotions from face pictures.
The dataset used for training is FER2013.

## Quick Start

1.Install dependencies

```bash
python -m pip install -r requirements.txt
```

2.Preprocess datasets

Preprocessing pipeline for AFEW and Emotic datasets. Refercences, download links and licences are given in the [Datasets](#datasets) section.
Datasets will be saved in /data/{split}_{dataset}.csv .

```bash
python -m src.data.preprocess
```

Options:

```bash
  -h, --help            show this help message and exit
  --dataset {afew,emotic,all}
                        Dataset to preprocess
  --data_dir DATA_DIR   Root path to input dataset directory
  --output_dir OUTPUT_DIR
                        Path to output processed CSVs
  --image_size IMAGE_SIZE
                        Output image size (width & height)
  --include_extra       Include EMOTIC extra training data
```

3.Train a single model

Pipeline for training models from /models on datasets in /data.
When saved, the weights which gave the best performances can be found in /output/{model}/seed{seed}_dataset-{dataset}_{dataaug if data augmentation enabled}_criterion-{criterion}_{CCCweight{CCC weight} if criterion = "combined"}_V{V weight}_A{A weight}_D{D weight}_opt-{optimizer}_lr{learning}_backboneLRscale{backbone learning rate scale}_bs{batch size}_dropout{dropout rate}/best_model_state.pth .

```bash
python -m src.training.train
```

Options:

```bash
  -h, --help            show this help message and exit
  --seed SEED           Random seed for reproducibility (default: 42)
  --dataset {fer,caers,afew}
                        Dataset to use for training and evaluation (default: fer)
  --input_size INPUT_SIZE
                        Image spatial resolution (default: 112)
  --num_workers NUM_WORKERS
                        DataLoader subprocess workers (default: 4)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs with no improvement after which training will be stopped (default: 20)
  --output_dir OUTPUT_DIR
                        Directory to save logs and model checkpoints (default: ./output)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --epochs EPOCHS       Number of training epochs (default: 100)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 1e-4)
  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimizer (default: 5e-4)
  --model {vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,efficientnet,mobilenet,mobilefacenet}
                        Model architecture to use (default: vgg16)
  --pretrained          Use pre-trained weights (default: False)
  --freezed             Freeze the convolutional layers of the model (default: False)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the regression head (default: 0.5)
  --optimizer {adam,sgd,adamw}
                        Optimizer to use for training (default: sgd)
  --device {cuda,cpu}   Device to use for training (default: cuda)
  --grad_clip GRAD_CLIP
                        Gradient clipping value (default: 0.0, no clipping)
  --data_augmentation   Apply data augmentation during training (default: False)
  --resume              Resume training from a previous checkpoint (default: False)
  --no_checkpoint       Do not save checkpoints (default: False)
  --no_model_save       Do not save the model (default: False)
  --criterion {mse,ccc,combined}
                        Loss function to use for training (default: mse)
  --VAD_weights VAD_WEIGHTS
                        Weights for the VAD loss (default: [1.0, 1.0, 1.0])
  --orth_loss_weight ORTH_LOSS_WEIGHT
                        Weight for the orthogonal loss (default: 0.5)
  --ccc_weight CCC_WEIGHT
                        Weight for the CCC loss (default: 0.5)
  --lr_factor LR_FACTOR
                        Factor by which the learning rate will be reduced (default: 0.1)
  --lr_patience LR_PATIENCE
                        Number of epochs with no improvement after which the learning rate will be reduced (default: 10)
  --lr_threshold LR_THRESHOLD
                        Minimum change in the monitored quantity to qualify as an improvement (default: 1e-4)
  --lr_threshold_mode LR_THRESHOLD_MODE
                        Mode to compare the monitored quantity to the threshold (default: rel)
  --lr_cooldown LR_COOLDOWN
                        Number of epochs to wait before resuming normal operation after a reduction in the learning rate (default: 0)
  --lr_min LR_MIN       Lower bound on the learning rate (default: 0.0)
```

4.Evaluate a model

Script to evaluate models trained with the pipeline above on the test dataset.

```bash  
python -m src.evaluation.test
```

Options:

```bash  
  -h, --help            show this help message and exit
  --input-size INPUT_SIZE
                        Image spatial resolution (default: 48).
  --device {cuda,cpu}   Device to use for evaluation (default: cuda if available, otherwise cpu).
  --split {Test,Val,Train}
                        Data split to evaluate on (default: Test).
  --state-dict-path STATE_DICT_PATH
                        Path to state dict with weights (.pth).
```

5.Plot a loss curve

Plots the curves corresponding to all log.csv file located in the specified directory.

```bash
python -m src.evaluation.plot_loss_curve
```

Options:

```bash
  -h, --help  show this help message and exit
  --dir DIR   Directory containing log.csv files.
```

6.Run hyperparameter optimization using optuna

*Work in Progress*

7.Analyze HPO results

Analyze Optuna Trial CSV Logs and displays :

- Best configurations
- Categorical and discrete parameter summary
- Statistical parameter importance and parameter recommandations
- Prune propensity analysis (parameters triggering pruning)
- Joint VAD weight vector analysis

```bash
python -m src.evaluation.analyze_optuna_log
```

Options:

```bash
  -h, --help            show this help message and exit
  --csv_path [CSV_PATH]
                        Optional path to trial CSV file
  --search_dir SEARCH_DIR
                        Directory to search if csv_path is not specified
  --top_k TOP_K         Number of top trials to display
  --save                Save text report of the analysis output
  --output_dir OUTPUT_DIR
                        Directory where saved report files are stored
```

## Repository Layout

- `data/`: untracked, store the dataset csvs
- `models/`: model architectures for emotion analysis
- `output/`: training logs and model weights
- `src/data/`: data preparation and preprocessing
- `src/evaluation/`: evaluation scripts
- `src/training/`: training scripts
- `src/transforms/`: transforms methods for data loading and data augmentation
- `src/utils/`: shared dataset and utility modules

## Datasets

### FER2013 re-labelled

- Source: Huo, Y., & Ge, Y. (2025). *VAD-Net: Multidimensional Facial Expression Recognition in Intelligent Education System.*
- Available at: [https://github.com/YeeHoran/VAD-Net/](https://github.com/YeeHoran/VAD-Net/)
- Licence: MIT License

### AFEW-VA

- Sources:
Kossaifi, J., Tzimiropoulos, G., Todorovic, S., & Pantic, M. (2017). AFEW-VA database for valence and arousal estimation in-the-wild, In *Image and Vision Computing*
Dhall, A., Goecke, R., Lucey, S., & Gedeon, T. (2012). Collecting Large, Richly Annotated Facial-Expression Databases from Movies, In *IEEE MultiMedia*, (vol. 19, no. 3, pp. 34-41)
- Available at: [https://ibug.doc.ic.ac.uk/resources/afew-va-database/](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
- License: Available for research purpose only
