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
In recent years, urban traffic congestion has become increasingly serious. This is particularly due to the dynamic nature, complexity, and unpredictability of the traffic environment, making Traffic Signal Control (TSC) a highly challenging issue.

In this study, we will use deep reinforcement learning to address the problem of traffic signal control. 

The research focuses on a single intersection, with the aim of first resolving the traffic issues at a single entry point. 
Through an in-depth study of a single intersection, we hope to master more effective traffic control strategies and apply these strategies to more complex traffic situations, thereby effectively alleviating urban traffic congestion.
# Problem Definition
● Objective: Minimize the queue length of vehicles​

● Study of traffic signal control under a single intersection scenario​

# Methodology
**Deep Q networks (DQN)​**


Deep Q-Networks (DQNs) are a type of neural network that is used to learn the optimal action-selection policy in a reinforcement learning setting.​

DQNs work by using a neural network to approximate the action-value function, which maps states of the environment to the expected return (i.e., the sum of future rewards) for each possible action. The goal of the DQN is to learn the optimal policy, which is the action that will maximize the expected return for each state.​

When we input the state to the agent (DQN), we hope it can decide how many seconds the traffic light should last based on the current traffic situation, and that is action.​

We use the Simulation of Urban MObility (SUMO) to simulate this action and calculate the reward as well as the next state. ​
Data Collection and Analysis Result

## Data Collection and Analysis Result
Agent

State

Action

Reward

# Hyperparameter

● The total sum of the maximum queue lengths generated by each lane in this time orientation


queuelength.png
：!(/path/to/queuelength.png)

## Conclusion and Future work  
Increase the acceptable level of delay into reward design.​

More traffic flow information is necessary.​

Multi Agent Deep Reinforcement Learning may be useful in multi- intersection.​
## Reference  

