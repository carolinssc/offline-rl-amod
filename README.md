# Offline-RL for AMOD
Official implementation of [Learning to Control Autonomous Fleets from Observation via Offline Reinforcement Learning](https://arxiv.org/abs/2302.14833)

<img align="center" src="readme_fig.png" width="1100"/></td> <br/>

## Prerequisites

You will need to have a working IBM CPLEX installation. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)

To install all required dependencies, run
```
pip install -r requirements.txt
```

## Contents

* `src/algos/sac.py`: PyTorch implementation of Graph Neural Networks for SAC.
* `src/algos/CQL.py`: PyTorch implementation of Graph Neural Networks for CQL and Cal-CQL.
* `src/algos/reb_flow_solver.py`: thin wrapper around CPLEX formulation of the Minimum Rebalancing Cost problem.
* `src/envs/amod_env.py`: AMoD simulator.
* `src/cplex_mod/`: CPLEX formulation of Rebalancing and Matching problems.
* `src/misc/`: helper functions.
* `src/conf/`: config files with hyperparamter settings.
* `data/`: json files for the simulator of the cities.
* `saved_files/`: directory for saving results, logging, etc.
* `ckpt/`: model checkpoints.
* `Replaymemories/`: datasets for offline RL.

## Examples

To train an agent online, `main_SAC.py` accepts the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test            activates agent evaluation mode (default: False)
    --max_episodes    number of episodes (default: 10000)
    --max_steps       number of steps per episode (default: T=20)
    --hidden_size     node embedding dimension (default: 256)
    --no-cuda         disables CUDA training (default: True, i.e. run on CPU)
    --directory       defines directory where to log files (default: saved_files)
    --batch_size      defines the batch size (default: 100)
    --alpha           entropy coefficient (default: 0.3)
    --p_lr            Actor learning reate (default 1e-3)
    --q_lr            Critic learning rate (default: 1e-3)
    --checkpoint_path path where to log or load model checkpoints
    --city            which city to train on 
    --rew_scale       reward scaling (default 0.01, for SF 0.1)
    --critic_version  defined critic version to use (default: 4)

simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --json_tsetp    (default: 3)
```

To train an agent offline, `main_CQL.py` accepts the following arguments (additional to main_SAC):
```
    
model arguments:
    --memory_path     path, where the offline dataset is saved
    --min_q_weight    conservative coefficient (eta in paper)
    --samples_buffer  number of samples to take from the dataset (max 10000)
    --lagrange_tresh  lagrange treshhold tau for autonamtic tuning of eta 
    --st              whether to standardize data (default: False)
    --sc              whether to scale (max-min) the data (default: Fasle)     
```
We provide the hyperparamters for CQL under src/conf. 

**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```
### Training and simulating an agent online
We tested our algorithms for following cities, i.e. city_name = {shenzhen_downtown_west, nyc_brooklyn, san_francisco} with the corresponding checkpoint_name={SHZ, NYC, SF}

1. To train an agent online:
```
python main_SAC.py --city {city_name}
```
2. To evaluate a pretrained agent run the following:
```
python main_SAC.py --city {city_name} --test True --checkpoint_path SAC_{checkpoint_name}
```
e.g. to evaluate a pretrained agent on Shenzhen: 
```
python main_SAC.py --city shenzhen_downtown_west --test True --checkpoint_path SAC_SHZ

```
### Training and simulating an agent offline

1. To train an agent offline for follwing dataset: M (Medium), H (High), G (Greedy), E (Expert)
```
python main_CQL.py --city city_name --memory_path dataset_name
```
e.g. to train an agent offline on the medium dataset on Shenzhen: 
```
python main_CQL.py --city shenzhen_downtown_west --memory_path Replaymemory_shenzhen_downtown_west_M
```

2. To evaluate a pretrained agent run the following:
```
python main_CQL.py --city {city_name} --test True --checkpoint_path CQl_{dataset}_{city_name}
```
e.g.
```
python main_CQL.py --city shenzhen_downtown_west --test True --checkpoint_path CQL_M_SHZ
```

### Online fine-tuning 
For the online fine-tuning, we use the same hyperparameters as for online SAC with the exception that during training, we sample 25% of the batch from the offline dataset the Cal-CQL agent was trained on and 75% from the online replay buffer.

1. To train an agent with Cal-CQL 
```
python main_Cal_CQL.py --city city_name --memory_path dataset_name --enable_cql True
```
2. To fine-tune a pretrained agent online run the following (chose frome Cal_CQL_M_SHZ, Cal_CQL_M_NYC, Cal_CQL_M_SF): 
```
python main_Cal_CQL.py --city city_name --memory_path dataset_name --fine_tune True --load_ckpt pre_trained_checkpoint 
```
e.g.
```
python main_Cal_CQL.py --city shenzhen_downtown_west --memory_path Replaymemory_shenzhen_downtown_west_M --fine_tune True --load_ckpt Cal_CQL_M_SHZ 
```
## Credits
This work was conducted as a joint effort with [Daniele Gammelli*](https://scholar.google.com/citations?user=C9ZbB3cAAAAJ&hl=de&oi=sra), [Filipe Rodrigues'](http://fprodrigues.com/), [Francisco C. Pereira'](http://camara.scripts.mit.edu/home/), at Technical University of Denmark' and Stanford University*. 

----------
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at csasc@dtu.dk.