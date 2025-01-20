# SimGNN
## Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             1.1.0
torch-scatter     1.4.0
torch-sparse      0.4.3
torch-cluster     1.4.5
torch-geometric   1.3.2
torchvision       0.3.0
scikit-learn      0.20.0
```

## Model options
```
  --filters-1             INT         Number of filter in 1st GCN layer.       Default is 128.
  --filters-2             INT         Number of filter in 2nd GCN layer.       Default is 64. 
  --filters-3             INT         Number of filter in 3rd GCN layer.       Default is 32.
  --tensor-neurons        INT         Neurons in tensor network layer.         Default is 16.
  --bottle-neck-neurons   INT         Bottle neck layer neurons.               Default is 16.
  --bins                  INT         Number of histogram bins.                Default is 16.
  --batch-size            INT         Number of pairs processed per batch.     Default is 128. 
  --epochs                INT         Number of SimGNN training epochs.        Default is 5.
  --dropout               FLOAT       Dropout rate.                            Default is 0.5.
  --learning-rate         FLOAT       Learning rate.                           Default is 0.001.
  --weight-decay          FLOAT       Weight decay.                            Default is 10^-5.
  --histogram             BOOL        Include histogram features.              Default is False.
```
## How to run
<p align="justify">
The following commands learn a neural network and score on the test set. Training a SimGNN model on the default dataset.</p>

```
python src/main.py
```
<p align="center">
<img style="float: center;" src="simgnn_run.jpg">
</p>

Training a SimGNN model for a 100 epochs with a batch size of 512.
```
python src/main.py --epochs 100 --batch-size 512
```
Training a SimGNN with histogram features.
```
python src/main.py --histogram
```
Training a SimGNN with histogram features and a large bin number.
```
python src/main.py --histogram --bins 32
```
Increasing the learning rate and the dropout.
```
python src/main.py --learning-rate 0.01 --dropout 0.9
```
You can save the trained model by adding the `--save-path` parameter.
```
python src/main.py --save-path /path/to/model-name
```
Then you can load a pretrained model using the `--load-path` parameter; **note that the model will be used as-is, no training will be performed**.
```
python src/main.py --load-path /path/to/model-name
```
----------------------------------------------------------------------
