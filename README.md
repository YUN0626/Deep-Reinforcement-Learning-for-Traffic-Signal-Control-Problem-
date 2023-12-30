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

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/DQN.jpg)



Deep Q-Networks (DQNs) are a type of neural network that is used to learn the optimal action-selection policy in a reinforcement learning setting.​

DQNs work by using a neural network to approximate the action-value function, which maps states of the environment to the expected return (i.e., the sum of future rewards) for each possible action. The goal of the DQN is to learn the optimal policy, which is the action that will maximize the expected return for each state.​

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/ora-rl.jpg)


When we input the state to the agent (DQN), we hope it can decide how many seconds the traffic light should last based on the current traffic situation, and that is action.​

We use the Simulation of Urban MObility (SUMO) to simulate this action and calculate the reward as well as the next state. ​

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/sumo.jpg)

1.Microscopic simulation: SUMO employs a microscopic simulation approach, modeling individual vehicles and their interactions within the traffic network. 
This allows for a realistic representation of traffic dynamics, including vehicle acceleration, deceleration, and lane-changing behaviors.

2.Traffic Demand Modeling: SUMO enables users to specify traffic demand patterns, including the generation and distribution of trips throughout the network. 
This functionality allows for the simulation of traffic conditions under different scenarios and helps assess the impact of changes in demand on the transportation system.

3.Traffic Control and Management: The simulation platform supports the modeling of various traffic control measures such as traffic lights, stop signs, and priority rules.
It is a valuable tool for assessing the effectiveness of traffic signal timings and other control strategies in improving traffic flow.

![image](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/assets/154335462/374bb159-fd50-4742-bc42-c0174eb48321)


## Data Collection and Analysis Result

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/%E8%BB%8A%E6%B5%81%E9%87%8F%E5%9C%96.jpg)

● State : The total number of vehicles on all lanes entering the intersection and the current time orientation

● Action :Green Light Duration [20,25,35,45]

● Reward :The total sum of the maximum queue lengths generated by each lane in this time orientation

● Direction :east-west direction green lights, yellow lights, and the north-south direction green lights, yellow lights, turn left  with the yellow light duration fixed at 3 seconds. 

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/Direction.jpg)


# Hyperparameter

● The total sum of the maximum queue lengths generated by each lane in this time orientation
 ![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/Hyperparameter.jpg)


1.Number of training episodes: We set the training to 500 episodes based on references from other literature.

2.Minimum value of epsilon: Initially, the agent will output an action, but we may not necessarily choose this action. 
The reason is that in the early stages of training, the decision-making of the neural network may not be accurate. 
We hope to experience a variety of situations early in training to ensure the diversity of the training data. Our setting for epsilon starts at 0.9 and eventually decreases to 0.05.

3.Soft update is a method of updating network parameters.

## Conclusion and Future work  
● Average Travel Time(ATT):
 It is defined as the total travel time of all vehicles divided by the number of vehicles, formally expressed by the following equation.
 
 ![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/ATT.jpg)

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/travel%20time.png)


● Average Waiting Time(AWT):
The waiting time of a vehicle is defined as the time (in seconds) spent with a speed below 0.1m/s since the last time it was faster than 0.1m/s.
(basically, the waiting time of a vehicle is reset to 0 every time it moves).
![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/AWT.jpg)

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/waitingtime.png)

● Queue Length (QL):
The queue length of a lane is the total number of vehicles queuing on a lane. 
The queuing vehicles are those with a speed less than 0.1 m/s on the given lane(the total number of halting vehicles for the last time step on the given lane.A speed of less  than 0.1 m/s is considered a halt.)

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/QL.jpg)

![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/queuelength.png)


● Fixed approach
It is the simplest traffic control approach that uses fixed phase duration with fixedcycle length and fixed order .The duration of green phases is set to 35 s and the yellow phase duration is 3 s


![替代文字](https://github.com/YUN0626/Deep-Reinforcement-Learning-for-Traffic-Signal-Control-Problem-/blob/main/Figure/%E6%AF%94%E8%BC%83.jpg)


1.Since the study focus on just one intersection, If there is more traffic flow information for additional road sections,
studying the design of traffic signals at two intersections can be more consistent with the simulation of actual road sections, and the reward will be more significant.

2.In the future, it is possible to incorporate the level of congestion that is acceptable to drivers into the reward design.
Achieving a strategy that simultaneously improves traffic conditions and is convincing to the public.

3.Multi Agent Deep Reinforcement Learning may be useful in multi- intersection.​

√ Approach for multiple agents to collaboratively learn.

√ Each agent pursues individual goals, considering others.

√ Challenges: Coordinating actions for overall optimization.


## Reference  

