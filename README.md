
The main driver file is in mcmix/mdpRep.ipynb. 
    This reproduces all the plots for the paper.
    
mcmix/MixMDP.ipynb is a testbed notebook on a smaller gridworld, 
    for experiments where only one trial is conducted.
    
mcmix/clustering.py, mcmix/emalg.py, mcmix/subspace.py, mcmix/helpers.py 
    contain functions for the method and helper functions.
    
mcmix/matplotlibrc contains the style file, from Jonny Brooks-Bartlett 
    (https://gist.github.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea)
    
The folder core contains code for the environment, mdp class, and more helper functions. 
    Code for that kindly provided by David Bruns-Smith, 
    Model-Free and Model-Based Policy Evaluation when Causality is Uncertain, 
    (https://arxiv.org/abs/2204.00956)