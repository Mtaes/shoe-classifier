# shoe-classifier
Required packages are listed in the `requirements.txt` file. Also `Dockerfile` is included.

**All scripts in the `src` directory should be run from there.**

## Data preparation
To prepare and download data before training run `prepare_data.py`.

```bash
cd src
python3 prepare_data.py
```
Short data exploration is in the `src/dataset.ipynb` notebook.

## Task 1
To train first model run `train_model_1.py` script.
This file accepts arguments:

- `--lr <lr>` - learning rate (default: `1e-3`)
- `--batch <batch_size>` - batch size (default: `4096`)
- `--workers <num_workers>` - number of workers in the dataloader (default: `4`)
- `--epochs <num_epochs>` - number of the epochs (default: `200`)

Training and validation tensorboard logs and checkpoints are saved to `logs/classifier_1/version_X` in the root of the repository.

Below is command that was run during development. Obtained results on the test split are also presented.
```bash
cd src
python3 train_model_1.py
```
- Accuracy: 0.9983
- Precision: 0.9967
- Recall: 0.9977
- F1Score: 0.9972

## Task 2

To train second model run `train_model_2.py` script.

This file accepts arguments:

- `--ckpt <path>` - path to checkpoint from task 1
- `--lr <lr>` - learning rate (default: `1e-3`)
- `--batch <batch_size`> - batch size (default: `4096`)
- `--workers <num_workers>` - number of workers in the dataloader (default: `4`)
- `--epochs <num_epochs>` - number of epochs (default: `200`)

Training and validation tensorboard logs and checkpoints are saved to `logs/classifier_2/version_X` in the root of the repository.

Below is command that was run during development.

```bash
cd src
python3 train_model_2.py  --batch 1 --lr 1e-2 --ckpt ../logs/classifier_1/version_0/checkpoints/epoch\=197-step\=2178.ckpt
```

Obtained model recognised correctly all images from the test split. Results are presented in the `src/task_2.ipynb` notebook.

## Reproducibility
To improve reproducibility of the results manual seeds were used in the code.