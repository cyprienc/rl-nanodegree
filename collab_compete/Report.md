# Report: Collaboration and Competition

This report describes the implementation chosen to solve the Tennis multi-agent Unity environment.

## Learning Algorithm

The algorithm chosen to solve this environment is the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
with Twin Delayed Deep Deterministic (TD3) policy gradient. It also takes inspiration from 
Factored Multi-Agent Centralised Policy Gradients (FACMAC).

- Centralized Critics: the key element of the MADDPG algorithm is that instead of training each agent using 
  an independent Actor-Critic (AC) network, the critics are "centralized", meaning they are omniscient and take as input
  all the observations and actions instead of just the observation and actions of their respective agent. 
  With this trick, the environment becomes "stationary" from the point of view of the critics.
  
- Factorized Critic: the FACMAC algorithm factorizes the critics into a single critic using a non-linear 
function. Here, because our agents optimal strategy is to collaborate fully, we can use this approach, albeit in a much
  simpler way, by summing the Q-values of the agents.
  
- Clipped Double Q-learning: to reduce the overestimation bias observed in AC methods, 
  four AC networks are trained per agent. The local and target ones (like in Double DQN) but also twins of the formers.
  During the Critic update, instead of only taking the target critic as reference for the temporal difference, 
  the minimum Q-value (between the target critic and its twin) is taken. The local critics are trained using SGD but 
  only the local AC network (and not his twin) is optimized in the policy step. With the minimum Q-value taken in the 
  critics update, it avoids divergence of the AC network from "reality".
  All the target networks are updated via soft-update with weight `tau` set to `5e-3`.
  
- "Delayed" policy update: as noted in the TD3 paper, learning a policy from an inaccurate Q-value function leads 
  to a poor policy. So the frequency of policy update is reduced relative to the critic ones. The policies are updated 
  every four updates of the critics (hyper-parameter `update_policy_every`).
  
- Policy smoothing: actions in the target Q-value functions are noised using a clipped gaussian noise 
  (`policy_smoothing` equal to 0.2, clipped `clip` between -0.5 and 0.5). 
  This avoids the critics to exploit peaks in the rewards. 
  
- Random play: 500 episodes are played completely randomly at the start of the training to initialize the replay buffer.
This prevents from overfitting the first episodes of the real training. The replay buffer size is `1e6`.
   
For the Stochastic Gradient Descent, two 'Adam' optimizers are used, one for the critic update and one actor policy, the
learning rate is 1e-3 and the other parameters of the optimizers are kept to the default values. Note that the optimizer
are not updating the same set of parameters.

For the exploration, a gaussian noise is added to the actions, the initial standard deviation 'sigma' is set to 0.2.
It decays with a weight 'sigma_scale' of 0.9999 and with a minimal noise `min_sigma` of 0.01.

Other hyper-parameters include:

- The discount factor gamma is 0.95.
- The maximum number of steps per episode is 10,000.
- The batch size is 64.
- 

## Model Architecture

As mentioned in the 'Learning Algorithm' section above. Multiple AC networks are used, all sharing the same architecture. 
The neural networks used to solve the environment contain 2 blocks:

- The 'actor' block: this block is composed of a batch normalization layer, followed by a dense layer 
  with relu activation and a subsequent batch normalization one.
  Another ReLu dense layer is followed by a final dense layer of the size of the action space with 
  Tanh activation to ensure the policy stays within the bounds of the action space.

- The 'critic' block: this block is constructed in the same way as the actor block, but the block instead terminates 
  with a single unit linear layer representing the (action,space) value.
  
All the hidden layers are 2 times the size of the input to the block.

## Results

The environment is solved after 1958 episodes:

![image](https://user-images.githubusercontent.com/5805228/124350743-59c95c00-dbee-11eb-9722-033fdaccd5f8.png)

## Ideas for Future Work

Next step would be to implement this algorithm with policy ensemble instead of learning a single policy per agent. 
Instead of using the sum as a factorization function for the centralized critic, a non-linear function could be used. 