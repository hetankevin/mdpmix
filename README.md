# Learning Mixtures of Markov Chains and MDPs

This repository contains the code for the paper Learning Mixtures of Markov Chains and MDPs,
    by Chinmaya Kausik, Kevin Tan and Ambuj Tewari. 
This paper was accepted to ICML 2023 for a short live presentation,
    and is available on arXiv at https://arxiv.org/abs/2211.09403.

The main driver file is in mcmix/mdpRep.ipynb. 
    This reproduces all the plots for the paper.
There is a supplementary driver file, mcmix/artistsMC.ipynb, 
    for an additional experiment 
    that will appear in the camera-ready version
    on the LastFM dataset.
That dataset is available here http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
    and what you need from the artist tags is in this repository.

    
mcmix/MixMDP.ipynb is a testbed notebook on a smaller gridworld, 
    for experiments where only one trial is conducted.
    
mcmix/clustering.py, mcmix/emalg.py, mcmix/subspace.py, mcmix/helpers.py 
    contain functions for the method and helper functions.
    
mcmix/matplotlibrc contains the style file, from Jonny Brooks-Bartlett 
    (https://gist.github.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea)
    
The folder core contains code for the environment, mdp class, and more helper functions. 
    The code for this section was kindly provided by David Bruns-Smith, 
    Model-Free and Model-Based Policy Evaluation when Causality is Uncertain, 
    (https://arxiv.org/abs/2204.00956)
