# Privacy-Preserving Multi-Target Multi-Domain Recommender Systems with Assisted AutoEncoders
This is an implementation of Privacy-Preserving Multi-Target Multi-Domain Recommender Systems with Assisted AutoEncoders
 
## Requirements
See requirements.txt

## Instruction
 - Global hyperparameters are configured in config.yml
 - Experimental setup are listed in make.py 
 - Hyperparameters can be found at process_control() in utils.py 
 - organization.py define local initialization, learning, and inference of one organization
 - assist.py demonstrate Multi-Target Assisted Learning (MTAL) algorithm
    - make_dataset() compute and distribute the pseudo-residual to all organizations
    - update() gather other domains' output and compute gradient assisted learning rate and gradient assistance weights
 - The data are split at split_dataset() in data.py

## Examples
 - Train Joint (ML100K, User-Aligned, Explicit Feedback, MF, without side information)
    ```ruby
    python train_recsys_joint.py --control_name ML100K_user_explicit_mf_0
    ```
 - Test Alone (ML1M, Item-Aligned, Implicit Feedback, NCF, with side information, 'Uniform' data partition)
    ```ruby
    python test_recsys_alone.py --control_name ML1M_item_implicit_nmf_1_random-8
    ```
 - Train MTAL (ML10M, User-Aligned, Explicit Feedback, AAE, without side information, 'Genre' data partition, with gradient assisted learning rate <img src="https://latex.codecogs.com/gif.latex?\eta_k=0.3"/>, without gradient assistance weights)
    ```ruby
    python train_recsys_assist.py --control_name ML10M_user_explicit_ae_0_genre_constant-0.3_constant
    ```
 - Test MTAL (Amazon, User-Aligned, Implicit Feedback, AAE, without side information, 'Genre' data partition, with gradient assisted learning rate <img src="https://latex.codecogs.com/gif.latex?\eta_k=0.1"/>, with gradient assistance weights, with partial alignment, with privacy enhancement (DP-10))
    ```ruby
    python test_recsys_assist.py --control_name Amazon_user_implicit_ae_0_genre_constant-0.1_optim_1_dp-10
    ```
