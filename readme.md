# MBPO PyTorch
A PyTorch reimplementation of MBPO (When to trust your model: model-based policy optimization)

# Dependency

Please refer to ./requirements.txt.

# Usage

    pip install -e .
    
    # default hyperparams in ./configs/mbpo.yaml
    python ./mbpo_pytorch/scripts/run_mbpo.py
    
    # you can also overwrite hyperparams by passing args, e.g.
    python ./mbpo_pytorch/scripts/run_mbpo.py --set seed=0 verbose=1 device="'cuda:0'" env.env_name='FixedHopper'

  
# Credits
1. [vitchyr/rlkit](https://github.com/vitchyr/rlkit)
2. [JannerM/mbpo](https://github.com/JannerM/mbpo)
3. [WilsonWangTHU/mbbl](https://github.com/WilsonWangTHU/mbbl)