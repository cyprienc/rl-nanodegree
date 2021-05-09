# Report: Navigation Project

This report describes the implementation chosen to solve the Banana Unity environment. 

## Learning Algorithm

The algorithm chosen to solve this environment is the Deep Q-learning algorithm (DQN). 
The following improvements were implemented:
-   Dueling architecture: 
        Instead of using a neural network directly predicting the state-action values Q directly, 
        the network architecture used to compute the Q values is split into two estimators: 
        One for the state value function V and one for the "advantage" function A. 
        This advantage function represents the difference between the Q values and the state value for a given state.
        In other words, it's the relative advantage to picking an action over the other possible actions. 
        To allow the network to differentiate between the V value and the advantage values, 
        the advantage estimator output is centered around its mean.
        
```python
q = value + (advantages - advantages.mean(1, True))
```

- Double DQN (DDQN) with soft-updates:
        To reduce the over-estimation bias in the state-action values caused by the max operation 
        in the Q-learning update, a common approach is to use a different network to estimate the expected
        value of the next state under the current policy. We first use the current policy to pick the next action
        that would be taken under the current policy. And then compute the expected state-action values using the 
        secondary "target" network. The updates of the second network are soft in the sense that the weights are 
        updated by taking a weighted average of the two networks, the averaging weight of the current network is the 
        hyperparameter 'tau' and '1 - tau' for the target network. The soft-updates are realized with 'tau' set to 1e-3.
        And the update is applied every 4 steps.
  
```python
next_actions = self.qnetwork_local(next_states).max(dim=1, keepdim=True)[1]
y = (
    rewards
    + (1 - dones)
    * gamma
    * self.qnetwork_target(next_states).gather(1, next_actions)
)
```

For the Stochastic Gradient Descent, a mean-squared error loss is used with the 'Adam' optimizer, 
the learning rate is 5e-4 and the other parameters of the optimizer are kept to the default values. 
The batch size is 64. 

Other hyperparameters include:
- The replay buffer side is 10,000.
- The discount factor gamma is 0.99.
- The epsilon starts at 1 and decays after every episode with a factor 0.99 until the minimal value of 0.01 is reached.
- The maximum number of steps per episode is 1000.

## Model Architecture

As mentioned in the 'Learning Algorithm' section above. A 'Dueling Architecture' is used. This means
that the neural network used to approximate the Q function contains 3 blocks:
- The 'feature map' block: this first, 2-layers-deep block, transforms the input state into features used in by 
  the subsequent blocks. The next two blocks are separated but share this block's outputs as inputs. 
  Each layer contains 5 times the size of the input state in cells with ReLu activations.

- The 'advantage function' block: this block is composed of a linear dense layer of the same size as the action space. 
  The output is centered around its mean to ensure the third block learns to estimate the value of the state.
  
- The 'value function' block: a simple linear block with one cell that computes the value of given the state.

The outputs of the last 2 block are summed to obtain the action-state values. 

## Results

The environment is solved after ~450 episodes:

![results](https://user-images.githubusercontent.com/5805228/117455292-f51bb900-af3e-11eb-9fcc-845c9225973f.png)

Here is the final agent playing:

![agent](https://i.imgur.com/BrTnwFp.gif)


## Ideas for Future Work

The prioritized replay buffer is an appealing idea to add to this agent. 
I had started creating a MaxSumTree to keep the max priority and sum of priorities: 
https://gist.github.com/cyprienc/5b5909799af150710360eefd5c473568

But efficient sampling is tricky, one good implementation is given by OpenAI: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py#L71

It utilizes Segment Trees to achieve efficient sampling.


TODO: Include gif
