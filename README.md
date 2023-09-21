# Disentangled Slot Attention
Repository for Disentangled Slot Attention (DISA).
<br><br>

## Prepare datasets
Use the commands below to download and read the TFRecord of a dataset. Replace ```RECORD_PATH``` with the path to the directory where you want to download the TFRecord, and ```DATA_PATH``` with the path to the directory where you want to store the dataset.
#### Tetrominoes
```bash
wget https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords -P RECORD_PATH
python data/tetrominoes.py --tfrecord_path RECORD_PATH --data_path DATA_PATH
```

#### Multi-dSprites
```bash
wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords -P RECORD_PATH
python data/multidsprites.py --tfrecord_path RECORD_PATH --data_path DATA_PATH
```

#### CLEVR
```bash
wget https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords -P RECORD_PATH
python data/clevr.py --tfrecord_path RECORD_PATH --data_path DATA_PATH
```

#### CLEVR6
To filter CLEVR into CLEVR6 (maximum of 6 objects + the background), replace ```CLEVR_PATH``` with the path to the directory where the CLEVR dataset is already stored, and replace ```CLEVR6_PATH``` with the path to the directory where you want to store the filtered CLEVR6 dataset.
```bash
python data/clevr6.py --clevr_path CLEVR_PATH --clevr6_path CLEVR6_PATH
```
<br>

## Train
Before running the commands below, move to the DISA directory.
### Object discovery
Use the command below to train a model on the object discovery task. Replace ```CONFIG_NAME``` with the name of the desired configuration (configs/objdisc_configs.json) to run.
```bash
python -m training.train --config CONFIG_NAME
```
If you need to use Distributed Data Parallel (DDP), replace ```training.train``` with ```training.train_dist``` and add the key ```num_gpus``` set to the desired integer in the configuration file.

### Property prediction
Use the command below to run the property prediction task. Replace ```CONFIG_NAME``` with the name of the desired configuration (configs/proppred_configs.json) to run.
```bash
python -m training.prop_pred --config CONFIG_NAME
```
<br>

## Evaluate
Before running the commands below, move to the DISA directory.
### Object discovery
Use the command below to evaluate a model on the object discovery task (BG-ARI, FG-ARI, MSE). Replace ```CONFIG_NAME``` with the name of the desired configuration (configs/objdisc_configs.json) to run, and ```CKPT_NAME``` with the name of the checkpoint to load (without .ckpt at the end).
```bash
python -m evaluation.obj_disc --config CONFIG_NAME --init_ckpt --CKPT_NAME
```
If you evaluate a model that samples initial slots and/or position/scale embeddings, add the key ```reps``` and set it to, e.g., 10 in order to evaluate each image in the test set 10 times.

### Property prediction
Use the command below to evaluate a configuration on the property prediction task. Replace ```CONFIG_NAME``` with the name of the desired configuration (configs/proppred_configs.json) to run.
```bash
python -m evaluation.prop_pred --config CONFIG_NAME
```
