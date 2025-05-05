A Pytorch implementation of the paper *MB-UDF: A Self-Supervised Model for Continuous Representation of Seafloor Topography Using Multibeam Echo Sounder Data*

## Preparation
1. Clone the repository;

2. Install the required packages:

+ python dependencies
```bash
cd MB-UDF
conda env create -f environment.yaml
```

+ C++ dependencies
```bash
cd extensions/chamfer_dist
python setup.py install
```

## Running
To run the demo, you can use the following command in root directory:
```bash
python run.py --gpu 0 --conf confs/base.conf --dataname <name_of_the_data> --dir <name_of_the_data>
```
If it's the first time you run the code, it will generate the `sample` point cloud (refer to the $S$ in the paper); then, re-run the command to have the model trained. Meshes are generated automatically after training, and the results are saved in the `./output` path.
 `Wandb` is used for logging the training process. 