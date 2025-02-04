# Wild-Time: A Benchmark of in-the-Wild Distribution Shifts over Time

## Overview
Distribution shift occurs when the test distribution differs from the training distribution, and it can considerably degrade performance of machine learning models deployed in the real world. Temporal shifts -- distribution shifts arising from the passage of time -- often occur gradually and have the additional structure of timestamp metadata. By leveraging timestamp metadata, models can potentially learn from trends in past distribution shifts and extrapolate into the future. While recent works have studied distribution shifts, temporal shifts remain underexplored. To address this gap, we curate Wild-Time, a benchmark of 5 datasets that reflect temporal distribution shifts arising in a variety of real-world applications, including patient prognosis and news classification.

![Wild-Time -- Dataset Description](data_description.png)

This repo includes scripts to download the Wild-Time datasets, code for baselines, and scripts for training and evaluating these baselines on Wild-Time.

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{yao2022wild,
  title={Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time},
  author={Yao, Huaxiu and Choi, Caroline and Cao, Bochuan and Lee, Yoonho and Koh, Pang Wei and Finn, Chelsea},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

We will release the arXiv version of our paper, along with the final code repository, in 1-2 months.

## Installation
To use our code, you first need to install your own version of pytorch, with version > 1.7.1 .

Then, you can use pip to install wildtime directly:

```
pip install wildtime==1.1.3
```
If you want to run a baseline test, please create a folder named `checkpoints` in your working directory.

### Accessing the MIMIC-IV dataset

1. Become a credentialed user on PhysioNet. This means that you must formally submit your personal details for review, so that PhysioNet can confirm your identity.
  - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
  - Follow these [instructions](https://physionet.org/credential-application/) for credentialing on PhysioNet.
  - Complete the "Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement.
  - [Log in](https://physionet.org/login/) to your PhysioNet account.
  - Go to the MIMIC-IV dataset [project page](https://physionet.org/content/mimiciv/2.0/).
  - Locate the "Files" section in the project description.
  - Click through, read, and sign the Data Use Agreement (DUA).
3. Go to https://physionet.org/content/mimiciv/1.0/ and download the following CSV files from the "core" and "hosp" modules to `./Data`:
    - patients.csv
    - admissions.csv
    - diagnoses_icd.csv
    - procedures_icd.csv
4. Decompress the files and put them under `./Data`.

## Run the code

### Importing dependencies
To load the wildtime data, you first need to import the dependencies:
```
import argparse
from configs import configs
configs = argparse.Namespace(**configs)
```
Where configs are parameters that contain the imported dataset. Importing datasets requires specifying the following parameters:
```
'dataset': 'yearbook', # Name of the dataset that you want to load, choices=['arxiv', 'drug', 'huffpost', 'mimic', 'fmow', 'yearbook']
'device': 0,  # gpu id
'random_seed': 1,  # random seed value
```
### Loading Wild-Time datasets
You can then use the following code to load the dataset:
```
from WildTime import dataloader
data = dataloader.getdata(configs)
```
If you want to load data of type Group, please set the parameter 'is_group_data' to 'True', here is an example:
```
data = dataloader.getdata(configs, is_group_data=True)
```
### Running baselines

To train a baseline on a Wild-Time dataset and evaluate under Eval-Fix (default evaluation), use the code:
```
from WildTime import baseline_trainer
baseline_trainer.train(configs)
```
Specify parameters in the config as follows:

- Specify the dataset with `--dataset`.
  - [arxiv, drug, fmow, huffpost, mimic, yearbook]
  - For MIMIC, specify one of two prediction tasks (mortality and readmission) using`--prediction_type=mortality` or `--prediction_type=readmission`.
- Specify the baseline with `--method`.
- To run Eval-Fix, add the flag `--offline`.
  - Specify the ID/OOD split time step with `--split_time`.
- To run Eval-Stream, add the flag `--eval_next_timesteps`.
- Set the training batch size with `--mini_batch_size`.
- Set the number of training iterations with `--train_update_iters`.
- [Optional] If using a data directory or checkpoint directory other than `./Data` and `./checkpoints`, specify their paths with `--data_dir` and `--log_dir`. 

#### CORAL
- Set the number of groups (e.g., number of time windows) with `--num_groups`.
- Set the group size (e.g., length of each time window) with `--group_size`.
- Specify the weight of the CORAL loss with `--coral_lambda` (default: 1.0).
- Add `--non_overlapping` to sample from non-overlapping time windows.

#### GroupDRO
- Set the number of groups (e.g., number of time windows) with `--num_groups`.
- Set the group size (e.g., length of each time window) with `--group_size`.
- Add `--non_overlapping` to sample from non-overlapping time windows.

#### IRM
- Set the number of groups (e.g., number of time windows) with `--num_groups`.
- Set the group size (e.g., length of each time window) with `--group_size`.
- Specify the weight of the IRM penalty loss with `--irm_lambda` (default: 1.0)
- Specify the number of iterations after which to anneal the IRM penalty loss wtih `--irm_penalty_anneal_iters` (default: 0).

#### ERM

#### LISA
- Specify the interpolation ratio $\lambda \sim Beta(\alpha, \alpha)$ with `--mix_alpha` (default: 2.0).

#### Mixup
- Specify the interpolation ratio $\lambda \sim Beta(\alpha, \alpha)$ with `--mix_alpha` (default: 2.0).

#### Averaged Gradient Episodic Memory (A-GEM)
- Set the buffer size with `--buffer_size` (default: 1000).

#### (Online) Elastic Weight Consolidation (EWC)
- Set the regularization strength (e.g., weight of the EWC loss) with `ewc_lambda` (default: 1.0). 

#### Fine-tuning (FT)

#### Synaptic Intelligence (SI)
- Set the SI regularization strength with `--si_c` (default: 0.1).
- Set the dampening parameter with `--epsilon` (default: 0.001).

#### SimCLR
- Specify the number of iterations for which to learn representations using SimCLR with `--train_update_iter`.
- Specify the number of iterations to finetune the classifier with `--finetune_iter`.

#### SwaV
- Specify the number of iterations for which to learn representations using SimCLR with `--train_update_iter`.
- Specify the number of iterations to finetune the classifier with `--finetune_iter`.

#### Stochastic Weighted Averaging (SWA)


## Scripts
In `configs/`, we provide a set of configs that can be used to train and evaluate models on the Wild-Time datasets. These scripts contain the hyperparameter settings used to benchmark the baselines in our paper.

All Eval-Fix scripts can be found located under `configs/eval-fix`. All Eval-Stream scripts are located under under `configs/eval-stream`.

## Checkpoints
For your reference, we provide some checkpoints for baselines used in our paper under the Eval-Fix setting. Please download the checkpoints [here](https://drive.google.com/drive/folders/1h_pvX4mhzVEddxenP-RkFl8yjKbl8FUN?usp=sharing) and put them under `model_checkpoints/`.

To use these checkpoints, add the flags `--load_model --log_dir=./model_checkpoints` to your command.

## Licenses
All code for Wild-Time is available under an open-source MIT license. We list the licenses for each Wild-Time dataset below:

- Yearbook: MIT License
- FMoW: [The Functional Map of the World Challenge Public License](https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE)
- MIMIC-IV (Readmission and Mortality): [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciv/view-license/0.4/)
- Drug-BA: MIT License
- Huffpost: CC0: Public Domain
- arXiv: CC0: Public Domain


## Acknowledgements

We thank the authors of all baselines. Most of our implementations follow the official implementations. We thank Zhenbang Wu for his assistance in preprocessing the MIMIC-IV dataset.
