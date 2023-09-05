# Disentangled Slot Attention
Repository for Disentangled Slot Attention (DISA).

## Prepare datasets
Use the commands below to download and read the TFRecord of a dataset. Replace ```RECORD_PATH``` with the path to the directory where you want to download the TFRecord, and ```DATA_PATH``` with the path to the directory where you want to store the dataset.
#### Tetrominoes
```bash
wget https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords -P RECORD_PATH
python data/tetrominoes.py RECORD_PATH DATA_PATH
```

#### Multi-dSprites
```bash
wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords -P RECORD_PATH
python data/multidsprites.py RECORD_PATH DATA_PATH
```

#### CLEVR
```bash
wget https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords -P RECORD_PATH
python data/clevr.py RECORD_PATH DATA_PATH
```

#### CLEVR6
To filter CLEVR into CLEVR6 (maximum of 6 objects + the background), replace ```CLEVR_PATH``` with the path to the directory where the CLEVR dataset is already stored, and replace ```CLEVR6_PATH``` with the path to the directory where you want to store the filtered CLEVR6 dataset.
```bash
python data/clevr6.py CLEVR_PATH CLEVR6_PATH
```
