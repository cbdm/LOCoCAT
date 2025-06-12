# LOCoCAT

This repository contains the files needed to reproduce the experiments in the LOCoCAT (Low-Overhead Classification of CAN bus Attack Types) project.
The goal is to use CAN bus data that has been corrupted due to a malicious intruder and classify what type of attack the intruder is executing.
Some of these scripts can be helpful if you intend to expand or branch off from this work or simply to replicate results.

## Getting started

First thing you need to do is to clone this repo.
Keep in mind that one of the datasets we use ([SynCAN](https://github.com/etas/SynCAN)) is added here as a submodule, so you need to clone this repo with an extra flag:
```
git clone --recurse-submodules https://github.com/cbdm/LOCoCAT.git
```

If you have already cloned it without the `--recurse-submodules` flag, you can try executing:
```
git submodule update --init --recursive
```
Alternatively, you can delete and clone it again.

The other four datasets should be downloaded from:
- https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
- https://ocslab.hksecurity.net/Datasets/survival-ids
- https://bitbucket.org/brooke-lampe/can-ml/
    - Note that for the `can-ml` dataset, we only use the `post-attack-labeled` data, so you can delete the other subdirectories to save some storage
- the google drive link in the [CAN-MIRGU repo](https://github.com/sampathrajapaksha/CAN-MIRGU)

Lastly, all steps/scripts described below assume you're running them from inside the `src` directory.
Nothing should break if you run them from a different directory, but the `out` folder might be created in some other root directory instead of this repo.

### Requirements

This repo heavily relies on scikit-learn, which is not supported out-of-the box on the m1 architecture at the time of this writing.
Because of that, I had to use conda to setup the environment and install dependencies.
Therefore, I've provided 3 files with a list of requirements.
- `pip_chill.txt`: this should work similarly to the usual `requirements.txt` in python repos. So your first try should be to run `pip install -r pip_chill.txt`. If that executes successfully, you should be able to run all scripts correctly.

If some of the dependencies do not work directly with pip in your system, you can use `conda_list.txt` and `pip_list.txt` to try to setup an environment like the one I had while developing. 

### Preparing the data

We have 5 datasets for the experiments:

1. SynCAN (https://github.com/etas/SynCAN)
2. Car-Hacking (https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
3. Survival-IDS (https://ocslab.hksecurity.net/Datasets/survival-ids)
4. can-ml (https://bitbucket.org/brooke-lampe/can-ml/)
5. CAN-MIRGU (https://github.com/sampathrajapaksha/CAN-MIRGU.git)

(1) is included as submodules inside the `data` subdir;
(2) and (3) need to be downloaded directly on their websites;
(4) must be cloned from the Bitbucket repository;
(5) must be downloaded from the google drive link in their github repository.

After downloading everything, you should have inside `data`:

- `SynCAN` directory;
- `9) Car-Hacking Dataset.zip` file;
- `20) Survival Analysis Dataset for automobile IDS.zip` file.
- `can-ml` directory;
- `CAN_MIRGU` directory;

Once you have all datasets downloaded and all dependencies installed, you can run `prepare_data.py --blocks` script to extract the attack data from the test sets and label their attack blocks.
This should generate 5 files inside the `data` subdir:

1. `syncan-blocks.csv` 
2. `car_hacking-blocks.csv`
3. `survival_ids-blocks.csv`
4. `can_ml-blocks.csv`
5. `can_mirgu-blocks.csv`

(1) has all the anomalous data from SynCAN, and each attack type is labeled with type and block number:

1. continuous
2. plateau
3. playback
4. suppress
5. flooding

(2) has all the anomalous data from Car-Hacking, and each attack is labeled with type and block number:

1. DoS
2. fuzzy
3. gear
4. rpm

(3) has all the anomalous data from Survival-IDS, and each attack is labeled with type and block number:

1. flooding
2. fuzzy
3. malfunction

(4) has (a subset of) the anomalous data from can-ml, and each attack is labeled with type and block number:

1. DoS
2. fuzzing
3. gear
4. interval
5. rpm
6. speed
7. standstill
8. systematic

(5) has all the anomalous data from CAN-MIRGU, and each attack is labeled with type and block number:

1. flam
2. injection every 0.02s
3. injection every 0.001s
4. masquerade
5. suspension

### Training models

**This and following sections have not been updated to include the new data yet. They show how to train and evaluate models for the first three datasets only.**

Once you have the prepared datasets, you can use the `train_models.py` script to do a grid search for the best parameters for classifier models.
You can choose which dataset(s) to use for training using the parameters in the script.

Instead of using all attack messages individually, in this project we group all messages from the same attack into a block.
The `dataset.py` module creates a `Dataset` class that makes it easier to create and process attack blocks from the csv file.
Using all data from SynCAN, you should have 545 attack blocks; using all data from Car-Hacking you should have 1200 attack blocks; using all data from Survival-IDS, you should have 29 attack blocks.

After we have these blocks, the `train_models.py` script will generate different featuresets by applying filters and calculating features that are defined in `filters.py` and `features.py`, respectively.
Then, for each of the featuresets, it will do a grid search to find the best parameters for each classifier model we want to train, and export (i) the test accuracy of each model into a csv file called `../out/[DATASET]/models/test_accuracy.csv` and (ii) objects with the trained models, filters, and features into a subdirectory called `../out/[DATASET]/models/[LIST-OF-FILTERS]`.
The models we are training are defined in `models.py`.

If you want to train different models, you can modify the appropriate modules (`models.py`/`filters.py`/`features.py`) and run the `train_models.py` script again.
You can also modify this latter script to export the new models to a different folder.

Keep in mind that the script will create subdirectories that are named based only on the filters.
So if you have different features with the same filters, the script might throw an assertion error.
If that's the case, you can manually change the "root subdir" in the script (e.g., `out2` instead of `out`).

### Measuring inference latency

Once you've finished training models with the script above, you should have a `../out/` directory that contains one subdir for each dataset with subdirectories for the filter-sets you have used (as described above).
To measure the latency of those models, you can use the `measure_latency.py` script.
This script receives 3 parameters as command line arguments:

- `--results_dir`: the path to the subdirectory where the models were saved (defaults to: `../out`)
- `--filename`: the name of the csv file you want to save the results to (defaults to: `inference_latency.csv`);
- `--n`: the number of times you want to run the inference model for each datapoint (defaults to: `1000`).

Using the default values, this script will find all dataset subdirectories in the `../out/` directory, then use all models inside the `models` subdir to predict the class for each attack block in the appropriate `dataset.pickle` file, and repeat this 1,000 times.
The results will then be saved to `../out/[DATASET]/models/inference_latency.csv`.

Keep in mind that due to limitations in the export process of features/filters with pickle, it might not be possible to measure latency in an architecture different than the one used for training.
If that happens, you can train the models again using the different architecture you want to measure latency on.

If you decide to run the previous script multiple times, you will end up with many `inference_latency.csv` files.
If all your files have the same prefix (e.g., you create `inference_latency-n=200.csv` and `inference_latency-n=400.csv`), you can use the `calculate_avg_latencies.py` script to accumulate all inference times and inference counts into a single file.

### Visualizing the data

The main results will be in (i) the test accuracies file that's exported by the `train_models.py` script, (ii) the size of the models in memory, and (iii) the average inference latency exported by the `calculate_avg_latencies.py` script.
The `draw_plots.py` script was used to generate the plots shown in the paper based on those results.

### LOCoCAT Paper results

The models, accuracies, and latencies measured for the paper can be found in the `out-pi0` subdir.
These results were obtained using a [Raspberry Pi Zero W](https://www.raspberrypi.com/products/raspberry-pi-zero-w/).

The related models trained for comparison can be found inside the `related_models` dir.

## Questions

Need help or have suggestions? Feel free to reach out!
