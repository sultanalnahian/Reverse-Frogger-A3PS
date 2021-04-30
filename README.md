# Reverse-Frogger-A3PS
This repository represents the implementation of [Influencing Reinforcement Learning through Natural Language Guidance](https://arxiv.org/abs/2104.01506). 

## Prerequisites:
1. Python 3.6
2. pytorch 1.2.0
3. **ml-agent-0.8.1:**

ml-agents-0.8.1 download link:
https://github.com/Unity-Technologies/ml-agents/tree/0.8.1

After downloading, install ml-agents-0.8.1 following the section **"Installing mlagents"** in the link: 
https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md

Some setup requirements for mlagents could be found: 
https://github.com/Unity-Technologies/ml-agents/blob/9602a8bef9f389964b1f5c1002217d02af54d191/ml-agents-envs/setup.py#L24

## Preprocessing:
After installing mlagents in ml-agents-0.8.1 folder, all codes from the **"ml-agents-0.8.1"** folder of the repo should be put inside of it. *ppo_rl_frogger.py* represents the coding for **Experience Driven agent** and *ppo+advice_rl_frog_max.py* is the code for A3PS architecture. The **"Advice generator"** folder contains the code for **Advice Driven agent**. 
 
 
 $ python ppo+advice_rl_frog_max.py
