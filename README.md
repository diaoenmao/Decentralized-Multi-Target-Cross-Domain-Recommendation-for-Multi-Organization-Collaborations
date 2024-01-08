# Decentralized Multi-Target Cross-Domain Recommendation for Multi-Organization Collaborations
[arXiv] This is an implementation of [Decentralized Multi-Target Cross-Domain Recommendation for Multi-Organization Collaborations](https://arxiv.org/abs/2110.13340)
- (a) User-Aligned (b) Item-Aligned Decentralized Multi-Target Cross-Domain Recommendation (DMTCDR) for Multi-Organization Collaborations. 
<p align="center">
<img src="/asset/dmtcdr.png">
</p>

- The Learning and Prediction stages of Multi-Target Assisted Learning (MTAL).
<p align="center">
<img src="/asset/mtal.png">
</p>

## Requirements
See requirements.txt

## Instruction
 - Global hyperparameters are configured in `config.yml`
 - Experimental setup are listed in `make.py`
 - Hyperparameters can be found at process_control() in `utils.py`
 - organization.py define local initialization, learning, and inference of one organization
 - assist.py demonstrate Multi-Target Assisted Learning (MTAL) algorithm
    - `make_dataset()` compute and distribute the pseudo-residual to all organizations
    - `update()` gather other domains' output and compute gradient assisted learning rate and gradient assistance weights
 - The data are split at `split_dataset()` in `data.py`

## Examples
 - Train Joint (ML1M, User-Aligned, Explicit Feedback, MF, without side information)
    ```ruby
    python train_recsys_joint.py --control_name ML1M_user_explicit_mf_0_genre_joint
    ```
 - Test Alone (ML1M, Item-Aligned, Implicit Feedback, NCF, with side information)
    ```ruby
    python test_recsys_alone.py --control_name ML1M_item_implicit_nmf_1_random-8_alone
    ```
- Train MDR (ML1M, User-Aligned, Explicit Feedback, MLP, without side information)
    ```ruby
    python train_recsys_mdr.py --control_name ML1M_user_explicit_mlp_0_genre_mdr
    ```
 - Train DMTMDR (Douban, User-Aligned, Explicit Feedback, AAE, without side information, with gradient assisted learning rate $\eta_k=0.3$, without gradient assistance weights)
    ```ruby
    python train_recsys_assist.py --control_name Douban_user_explicit_ae_0_genre_assist_constant-0.3_constant
    ```
 - Test DMTMDR (Amazon, User-Aligned, Implicit Feedback, AAE, without side information, with gradient assisted learning rate $eta_k=0.1$, with gradient assistance weights, with partial alignment (0.5), with privacy enhancement (DP-10))
    ```ruby
    python test_recsys_assist.py --control_name Amazon_user_implicit_ae_0_genre_assist_constant-0.1_optim_0.5_dp-10
    ```

## Results
- Results across all assistance rounds. Item-alignedDMTCDR outperforms 'Alone' baseline for both explicit andimplicit feedback.
<p align="center">
<img src="/asset/result.png">
</p>

## Acknowledgements
*Enmao Diao  
Jie Ding  
Vahid Tarokh*
