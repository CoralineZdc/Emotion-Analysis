# Emotion Analysis

This project includes a model training tool to train models to anlayse emotions from face pictures.
The dataset used for training is FER2013.

## Quick Start

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Preprocess datasets

Preprocessing pipeline for AFEW, Emotic and HECO datasets. Refercences, download links and licences are given in the [Datasets](#datasets) section.
Datasets will be saved in /data/{split}_{dataset}.csv .

```bash
python -m src.data.preprocess
```

Options:

```bash
  -h, --help            show this help message and exit
  --dataset {afew,emotic,heco,all}
                        Dataset to preprocess
  --data_dir DATA_DIR   Root path to input dataset directory
  --output_dir OUTPUT_DIR
                        Path to output processed CSVs
  --image_size IMAGE_SIZE
                        Output image size (width & height)
  --target_age {child,adult}
                        Filter samples by target age group
  --include_extra       Include EMOTIC extra training data
  --no_resplit          Disable forced-resplitting of dataset and retain original splits if available
```

### Visualize dataset VAD distribution

Visualize Valece, Arousal and eventually Dominance isolated and pairwise distribution across splits.

```bash
python -m src.data.visualize_vad
```

Options:

```bash
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing split CSVs
  --dataset {afew,emotic,heco}
                        Dataset to visualize
  --target_age {child,adult}
                        Age suffix filter if applicable
  --save_dir SAVE_DIR   Directory to save generated plot images
```

### Train a single model

Pipeline for training models from /models on datasets in /data.
When saved, the weights which gave the best performances can be found in /output/{model}/seed{seed}_dataset-{dataset}_{dataaug if data augmentation enabled}_criterion-{criterion}_{CCCweight{CCC weight} if criterion = "combined"}_V{V weight}_A{A weight}_D{D weight}_opt-{optimizer}_lr{learning}_backboneLRscale{backbone learning rate scale}_bs{batch size}_dropout{dropout rate}/best_model_state.pth .

```bash
python -m src.training.train
```

#### Execution parameters

```bash
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
                        Directory to save logs and checkpoints (default: outputs)
  --seed SEED           Random seed for reproducibility (default: 42)
  --device {cuda,cpu}   Device to use for training (default: cuda if available, otherwise cpu)
  --resume              Resume training from a previous checkpoint (default: False)
  --no_model_save       Do not save the model (default: False)
  --num_workers NUM_WORKERS
                        DataLoader subprocess workers (default: 4)
  --use_amp             Use Automatic Mixed Precision (AMP) training (default: True)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs with no improvement after which training will be stopped (default: 20)
  --epochs EPOCHS       Number of training epochs (default: 100)
```

#### Dataset parameters

```bash
  --dataset {fer,caers,afew,emotic,emotic-child,heco}
                        Dataset to use for training and evaluation (default: fer)
  --input_size INPUT_SIZE
                        Image spatial resolution (default: 112)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
```

#### Model definition parameters

```bash
  --model {vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,efficientnet,mobilenet,mobilefacenet}
  --pretrained          Use pre-trained weights (default: False)
  --weights_source {imagenet,custom}
                        Source of pre-trained weights (default: imagenet)
  --freezed             Freeze the convolutional layers of the model (default: False)
                        Model architecture to use (default: vgg16)
  --unfreeze_epoch UNFREEZE_EPOCH
  --head_lr HEAD_LR     Learning rate for the optimizer head (default: 1e-4)
  --backbone_lr BACKBONE_LR
                        Learning rate for the optimizer backbone (default: 1.0)
                        Epoch at which to unfreeze frozen backbone (-1 disables mid-training unfreeze)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the regression head (default: 0.5)
```

#### Loss criterion and backpropagation parameters

```bash
  --criterion {mse,ccc,combined}
                        Loss function to use for training (default: mse)
  --VAD_weights VAD_WEIGHTS
                        Weights for the VAD loss (default: "1.0, 1.0, 1.0")
  --ccc_weight CCC_WEIGHT
  --optimizer {adam,sgd,adamw}
                        Optimizer to use for training (default: sgd)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum norm for gradient clipping (default: 1.0, set 0 to disable)
```

#### Scheduler parameters

```bash
                        Weight for the CCC loss (default: 0.5)
  --scheduler {reduce_on_plateau,cosine_annealing}
                        Learning rate scheduler to use (default: reduce_on_plateau)
  --lr_factor LR_FACTOR
                        Factor by which the learning rate will be reduced (default: 0.1)
  --lr_patience LR_PATIENCE
                        Number of epochs with no improvement after which the learning rate will be reduced (default: 10)
```

#### Regularization parameters

```bash
  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimizer (default: 5e-4)
  --data_augmentation   Apply data augmentation during training (default: False)
```

### Evaluate a model

Script to evaluate models trained with the pipeline above on the test dataset.

```bash  
python -m src.evaluation.test
```

Options:

```bash  
  -h, --help            show this help message and exit
  --device {cuda,cpu}   Device to use for evaluation (default: cuda if available, otherwise cpu).
  --split {Test,Val,Train}
                        Data split to evaluate on (default: Test).
  --state_dict_dir STATE_DICT_DIR
                        Path from /output/ to dir containing the state dict with weights (.pth).
```

### Plot a loss curve

Plots the curves corresponding to all log.csv file located in the specified directory.

```bash
python -m src.evaluation.plot_loss_curve
```

Options:

```bash
  -h, --help  show this help message and exit
  --dir DIR   Directory containing log.csv files.
```

### Run hyperparameter optimization using optuna

Runs Hyperparameter Optimization with Optuna on desired model and dataset.

```bash
python -m src.training.optuna_search
```

#### Study definition parameters

```bash
  --study_name STUDY_NAME
  --storage STORAGE
  --log_dir LOG_DIR
  --resume_study        Resume an existing Optuna study if it exists
  --pruner {hyperband,median,none}
                        Pruner type for Optuna trials
```

#### Training parameter ranges

```bash
  --range_dataset RANGE_DATASET
                        Datasets to search over
  --range_model RANGE_MODEL
                        Backbone model options
  --range_head_lr RANGE_HEAD_LR
                        'Min, Max' bounds for head LR
  --range_backbone_lr RANGE_BACKBONE_LR
                        'Min, Max' bounds for backbone LR
  --range_weight_decay RANGE_WEIGHT_DECAY
                        'Min, Max' bounds for weight decay
  --range_unfreeze_epoch RANGE_UNFREEZE_EPOCH
                        'Min, Max' bounds for unfreeze epoch
  --range_input_size RANGE_INPUT_SIZE
                        Discrete input resolutions
  --range_batch_size RANGE_BATCH_SIZE
                        Discrete batch sizes
  --range_dropout_rate RANGE_DROPOUT_RATE
                        'Min, Max, Step' for dropout
  --range_weights_source RANGE_WEIGHTS_SOURCE
                        Weights source options
  --range_optimizer RANGE_OPTIMIZER
                        Optimizers to search over
  --range_criterion RANGE_CRITERION
                        Criterions to search over
  --range_ccc_weight RANGE_CCC_WEIGHT
                        'Min, Max, Step' for CCC loss weight
  --range_scheduler RANGE_SCHEDULER
                        Schedulers to search over
  --range_lr_factor RANGE_LR_FACTOR
                        Discrete learning rate decay factors
  --range_lr_patience RANGE_LR_PATIENCE
                        'Min, Max' bounds for scheduler patience
  --range_lr_warmup_epochs RANGE_LR_WARMUP_EPOCHS
                        'Min, Max' bounds for warmup epochs
```

#### Arguments for fixing specific training parameters

```bash
  --output_dir OUTPUT_DIR
                        Directory to save logs and checkpoints (default: outputs)
  --seed SEED           Random seed for reproducibility (default: 42)
  --device {cuda,cpu}   Device to use for training (default: cuda if available, otherwise cpu)
  --resume              Resume training from a previous checkpoint (default: False)
  --no_model_save       Do not save the model (default: False)
  --num_workers NUM_WORKERS
                        DataLoader subprocess workers (default: 4)
  --use_amp             Use Automatic Mixed Precision (AMP) training (default: True)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Number of epochs with no improvement after which training will be stopped (default: 20)
  --epochs EPOCHS       Number of training epochs (default: 100)

  --dataset {fer,caers,afew,emotic,emotic-child,heco}
                        Dataset to use for training and evaluation (default: fer)
  --input_size INPUT_SIZE
                        Image spatial resolution (default: 112)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)

  --model {vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,efficientnet,mobilenet,mobilefacenet}
  --pretrained          Use pre-trained weights (default: False)
  --weights_source {imagenet,custom}
                        Source of pre-trained weights (default: imagenet)
  --freezed             Freeze the convolutional layers of the model (default: False)
                        Model architecture to use (default: vgg16)
  --unfreeze_epoch UNFREEZE_EPOCH
  --head_lr HEAD_LR     Learning rate for the optimizer head (default: 1e-4)
  --backbone_lr BACKBONE_LR
                        Learning rate for the optimizer backbone (default: 1.0)
                        Epoch at which to unfreeze frozen backbone (-1 disables mid-training unfreeze)
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the regression head (default: 0.5)

  --criterion {mse,ccc,combined}
                        Loss function to use for training (default: mse)
  --VAD_weights VAD_WEIGHTS
                        Weights for the VAD loss (default: "1.0, 1.0, 1.0")
  --ccc_weight CCC_WEIGHT
  --optimizer {adam,sgd,adamw}
                        Optimizer to use for training (default: sgd)
  --max_grad_norm MAX_GRAD_NORM
                        Maximum norm for gradient clipping (default: 1.0, set 0 to disable)

                        Weight for the CCC loss (default: 0.5)
  --scheduler {reduce_on_plateau,cosine_annealing}
                        Learning rate scheduler to use (default: reduce_on_plateau)
  --lr_factor LR_FACTOR
                        Factor by which the learning rate will be reduced (default: 0.1)
  --lr_patience LR_PATIENCE
                        Number of epochs with no improvement after which the learning rate will be reduced (default: 10)

  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimizer (default: 5e-4)
  --data_augmentation   Apply data augmentation during training (default: False)
```

### Analyze HPO results

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
  --plot                Generate interactive CCC vs RMSE plot and save as HTML
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

- Source: Huo, Y., & Ge, Y. (2025). _VAD-Net: Multidimensional Facial Expression Recognition in Intelligent Education System._
- Available at: [https://github.com/YeeHoran/VAD-Net/](https://github.com/YeeHoran/VAD-Net/)
- Licence: MIT License

### AFEW-VA

- Sources:
Kossaifi, J., Tzimiropoulos, G., Todorovic, S., & Pantic, M. (2017). AFEW-VA database for valence and arousal estimation in-the-wild, In _Image and Vision Computing_
Dhall, A., Goecke, R., Lucey, S., & Gedeon, T. (2012). Collecting Large, Richly Annotated Facial-Expression Databases from Movies, In _IEEE MultiMedia_, (vol. 19, no. 3, pp. 34-41)
- Available at: [https://ibug.doc.ic.ac.uk/resources/afew-va-database/](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
- License: Available for research purpose only

### EMOTIC

- Sources:
Kosti, R., Alvarez, J. M., Recasens, A., & Lapedriza, A. (2020) Context Based Emotion Recognition Using EMOTIC Dataset, In _IEEE Transactions on Pattern Analysis and Machine Intelligence_, (vol. 42, no. 11, pp. 2755-2766)
Kosti, R., Alvarez, J. M., Recasens, A., & Lapedriza, A. (2017) Emotion Recognition in Context, In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, (pp. 1667-1675)
Kosti, R., Alvarez, J. M., Recasens, A., & Lapedriza, A. (2017) EMOTIC: Emotions in Context Dataset, In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops_, (pp. 61-69)
Kosti, R. (2019) _Visual scene context in emotion perception_. [https://hdl.handle.net/10803/667808](https://hdl.handle.net/10803/667808)
- Available at: [https://github.com/rkosti/emotic](https://github.com/rkosti/emotic)
- License: Available for non-commercial research purpose only

### HECO

- Source: Yang, D., Huang, S., Wang, S., Liu, Y., Zhai, P., Su, L., Li, M., & Zhang, L. (2022). Emotion Recognition for Multiple Context Awareness. In _Computer Vision - ECCV 2022_ (Vol. 13697, pp. 144–162).
- Available at: [https://heco2022.github.io/](https://heco2022.github.io/)
- License: Unspecified
