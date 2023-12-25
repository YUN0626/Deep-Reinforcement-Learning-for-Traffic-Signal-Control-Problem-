# Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-
December 25,2023/by 陳冠文、楊婷芸
# Content
● Background and Motivation

● Problem Definition

● Methodology

● Data Collection and Analysis Result

● Conclusion and Future work  

● Reference  
# Background and Motivation
In recent years, urban traffic congestion has become increasingly severe, especially considering the dynamic nature, complexity, and uncertainty of the traffic environment, making Traffic Signal Control (TSC) a highly challenging issue.

Here,we are going to use Deep Reinforcement Learning to solve Traffic Signal Control Problem
# Methodology
Deep Q networks (DQN)​
Deep Q-Networks (DQNs) are a type of neural network that is used to learn the optimal action-selection policy in a reinforcement learning setting.​

DQNs work by using a neural network to approximate the action-value function, which maps states of the environment to the expected return (i.e., the sum of future rewards) for each possible action. The goal of the DQN is to learn the optimal policy, which is the action that will maximize the expected return for each state.​

When we input the state to the agent (DQN), we hope it can decide how many seconds the traffic light should last based on the current traffic situation, and that is action.​

We use the Simulation of Urban MObility (SUMO) to simulate this action and calculate the reward as well as the next state. ​

Then, we store the (state, action, reward, next state) in the replay buffer as a piece of training data.​
