# MBPO PyTorch
A PyTorch reimplementation of MBPO (When to trust your model: model-based policy optimization)

# <font color=#A52A2A >Deprecated Warning</font>
The owner of this repo has graduated and this repo is no longer maintained. Please refer to this new [MBPO](https://github.com/x35f/model_based_rl) Pytorch re-implementation, which is a submodule of the [Unstable Baselines](https://github.com/x35f/unstable_Baselines) project maintained by researchers from the same [lab](http://www.lamda.nju.edu.cn/MainPage.ashx). This new MBPO re-implementation strictly follows the original TF implementation and has been tested on several MuJoCo tasks.

# Dependency

Please refer to ./requirements.txt.

# Usage

    pip install -e .

    # default hyperparams in ./configs/mbpo.yaml
    # remember to CHANGE proj_dir to your actual directory 
    python ./mbpo_pytorch/scripts/run_mbpo.py
    
    # you can also overwrite hyperparams by passing args, e.g.
    python ./mbpo_pytorch/scripts/run_mbpo.py --set seed=0 verbose=1 device="'cuda:0'" env.env_name='FixedHopper'

  
# Credits
1. [vitchyr/rlkit](https://github.com/vitchyr/rlkit)
2. [JannerM/mbpo](https://github.com/JannerM/mbpo)
3. [WilsonWangTHU/mbbl](https://github.com/WilsonWangTHU/mbbl)