# Reverse-Frogger-A3PS
This repository represents the implementation of [Influencing Reinforcement Learning through Natural Language Guidance](https://arxiv.org/abs/2104.01506). 

## Prerequisites:
1. Windows 10
2. Python 3.6
3. pytorch 1.2.0
4. **ml-agent-0.8.1:**

ml-agents-0.8.1 download link:
https://github.com/Unity-Technologies/ml-agents/tree/0.8.1

After downloading, install ml-agents-0.8.1 following the section **"Installing mlagents"** in the link: 
https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md

Some setup requirements for mlagents could be found: 
https://github.com/Unity-Technologies/ml-agents/blob/9602a8bef9f389964b1f5c1002217d02af54d191/ml-agents-envs/setup.py#L24

## Preprocessing:
After installing mlagents in ml-agents-0.8.1 folder, all codes from the **"ml-agents-0.8.1"** folder of the repo should be put inside of it.*experience_driven_agent.py* represents the coding for **Experience Driven agent** and *a3ps.py* is the code for A3PS architecture. The **"Advice generator"** folder contains the code for **Advice Driven agent**. 
 
 ## Training:
 As an example game environment, you can download the game from the link: https://drive.google.com/file/d/1eRX32aOssgBPYm2YOcXxAknP23XaHUe5/view?usp=sharing
 
 The downloaded windows build  is required to be put inside the ml-agents-0.8.1 folder. 
 
 To train the agent, run: 
 ```console
$ python ppo+advice_rl_frog_max.py
```
 
