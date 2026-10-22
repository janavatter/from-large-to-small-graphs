# From Large to Small Graphs: A Data Management Perspective on Sampling in Graph Neural Network Training
This repository contains the code and artifacts of our experiments.

## Artifacts
As datasets we use three datasets from the [OGB](https://ogb.stanford.edu/) collection, namely `arxiv`, `products`, and `papers100M`. In addition, we use eighteen datasets integrated in the [Deep Graph Library](https://www.dgl.ai/) (DGL): [`AmazonCoBuyComputer`](https://docs.dgl.ai/generated/dgl.data.AmazonCoBuyComputerDataset.html#dgl.data.AmazonCoBuyComputerDataset), [`AmazonCoBuyPhoto`](https://docs.dgl.ai/generated/dgl.data.AmazonCoBuyPhotoDataset.html#dgl.data.AmazonCoBuyPhotoDataset), [`Citeseer`](https://docs.dgl.ai/generated/dgl.data.CiteseerGraphDataset.html#dgl.data.CiteseerGraphDataset), [`Cora`](https://docs.dgl.ai/generated/dgl.data.CoraGraphDataset.html), [`Flickr`](https://docs.dgl.ai/generated/dgl.data.FlickrDataset.html), [`CoauthorCS`](https://docs.dgl.ai/generated/dgl.data.CoauthorCSDataset.html#dgl.data.CoauthorCSDataset), [`CoauthorPhysics`](https://docs.dgl.ai/generated/dgl.data.CoauthorPhysicsDataset.html#dgl.data.CoauthorPhysicsDataset), [`Reddit`](https://docs.dgl.ai/generated/dgl.data.RedditDataset.html), [`Pubmed`](https://docs.dgl.ai/generated/dgl.data.PubmedGraphDataset.html#dgl.data.PubmedGraphDataset), [`Yelp`](https://docs.dgl.ai/generated/dgl.data.YelpDataset.html), [`Chameleon`](https://docs.dgl.ai/generated/dgl.data.ChameleonDataset.html), [`Squirrel`](https://docs.dgl.ai/generated/dgl.data.SquirrelDataset.html), [`Actor`](https://docs.dgl.ai/generated/dgl.data.ActorDataset.html), and the [`HeterophilousGraphDataset`](https://docs.dgl.ai/generated/dgl.data.HeterophilousGraphDataset.html) datasets `RomanEmpire`, `AmazonRatings`, `Minesweeper`, `Tolokers`, and `Questions`.
All non-fairness datasets are downloaded automatically when first used.

For the fairness experiments, the `Credit Defaulter`, `German Credit`, and `Bail`/`Recidivism` graph datasets use the [NIFTY](https://github.com/chirag-agarwall/nifty) distribution; their preprocessing layout follows [EDITS](https://github.com/yushundong/EDITS/tree/main/dataset). The `Pokec-n` and `Pokec-z` datasets use the [FairGNN](https://github.com/WuGL-CS/FairGNN) distribution, which is sampled from soc-Pokec; their preprocessing reference is [FairGKD](https://github.com/ZzoomD/FairGKD/tree/master/datasets). These datasets are not included in this repository; download them from their original sources and place them under `dataset/`.

All experiments were logged via [Weights & Biases (W&B)](https://wandb.ai/site). The experiment configurations are described in a spreadsheet in the directory `parameters` and the results can be found in the directory `artifacts/`.

## Reproducibility and Experiments
All experiments can be reproduced using the parameters in the spreadsheet in `parameters`. The requirements are listed in `requirements.txt`. Training runs are initialized in W&B offline mode, so no W&B login is required. To upload a completed run, authenticate with `wandb login` and use `wandb sync`. Optionally, copy `config.example.json` to `config.json` and set `entity` to specify the W&B team or username used when syncing runs. For GPU experiments, CUDA needs to be installed.

Use Python 3.10 for the supported environment.

Start by cloning the repository and installing the requirements:

```bash
git clone https://github.com/janavatter/from-large-to-small-graphs.git
cd from-large-to-small-graphs
pip install -r requirements.txt
```

A sample command to run an experiment is:

```
python3 main.py --data ogbn-arxiv --epoch 20 --hidden 256 --sampler neighbor --layer 2 --fanout 10 --wandb sampling-study-base-experiments-arxiv --batchsize 1024 --mode cpu --seed 42
```

This will automatically download the needed dataset (might take some time), convert it to a DGL (Deep Graph Library) graph, and start the training process.
The optional `--seed` argument defaults to `0`; set it explicitly when reproducing a particular run. `--run` labels a repeated experiment but does not alter the seed.
Running `python3 main.py` uses the default CPU `ogbn-arxiv`/GCN/neighbor-sampling configuration with a fanout of 10 per layer.


## GPU-based training
If using GPU-based training, make sure that CUDA is installed. You can use the following commands:
```bash
sudo apt update
sudo apt install -y build-essential
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

echo -e "\nexport PATH=/usr/local/cuda-11.7/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

pip install torch torchvision torchaudio

pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Experiment Parameters
For the training, one needs to set some parameters. We give an overview of the parameters by explaining them and describing the possible values.

1. *data*: the name of the dataset, choose from (*ogbn-arxiv*, *ogbn-products*, *ogbn-papers100M*, *amazoncomputer*, *amazonphoto*, *citeseer*, *cora*, *coauthorcs*, *coauthorph*, *flickr*, *pubmed*, *reddit*, *yelp*, *credit*, *german*, *bail*, *pokec-n*, *pokec-z*, *chameleon*, *squirrel*, *actor*, *romanempire*, *amazonratings*, *minesweeper*, *tolokers*, *questions*).
2. *model*: the GNN model, choose from (gcn, gat, ggnn, appnp, sage, gin).
3. *sampler*: the sampler to use, choose from (*neighbor*, *vrgcn*, *fastgcn*, *ladies*, *labor*, *cluster*, *saint*, *shadow*, *none*).
4. *fanout*: the fanout of the sampling strategy, use with *neighbor*, *vrgcn*, *labor*, *shadow*.
5. *fastladies*: the nodes per layer, use with *fastgcn* and *ladies*.
6. *budget*: the budget for the *saint* sampler.
7. *num_parts*: the number of partitions for *cluster*.
8. *epoch*: the number of epochs.
9. *hidden*: the size of the hidden dimensions.
10. *layer*: the number of layers.
11. *batchsize*: the batchsize.
12. *mode*: use *cpu* for CPU training, *mixed* for CPU-GPU training, *puregpu* for GPU training.
13. *wandb*: the name of the wandb project.

The experiment parameters we used are described in a spreadsheet in the directory `parameters` (`GNNSamplingStudyExperimentParameters.xlsx`).

## Sampler Selector
The code and data for the sampler selector can be found in the directory `selector/`. Run the main pipeline from that directory:
```
cd selector
python3 sampler_selector.py -m objective=test_acc test_dataset=products use_advanced_features=true use_regression_approach=true normalize=true use_group_cv=true
```
with the following experiment parameters:
1. *objective*: the optimization objective, choose from (*test_acc*, *time_per_epoch*, *used_memory*)
2. *test_dataset*: what the test dataset should be (*ogbn-arxiv*, *ogbn-products*, *ogbn-papers100M*, *amazoncomputer*, *amazonphoto*, *citeseer*, *cora*, *coauthorcs*, *coauthorph*, *flickr*, *pubmed*, *reddit*, *yelp*).
3. *use_advanced_features*: whether advanced feature engineering should be used (bool)
4. *use_regression_approach*: indicates if the regression-based approach should be used (bool)
5. *normalize*: indicates if the features should be normalized (bool)
6. *use_group_cv*: whether leave-one-out cross-validation (LOOCV) should be used (bool, default: true)

To run the baselines, one can, in addition to the parameters above, set the seed via *seed=42*. This automatically runs the random, the majority-based, and the expert baselines.
```
cd selector
python3 sampler_selector_baselines.py -m objective=test_acc test_dataset=products seed=42
```
