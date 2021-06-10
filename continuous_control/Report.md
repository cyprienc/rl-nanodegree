# Report: Continuous Control Project

This report describes the implementation chosen to solve the multi-agent version of the Reacher Unity environment.

## Learning Algorithm

The algorithm chosen to solve this environment is the Deep Deterministic Policy Gradient (DDPG). As in the original
paper, a few techniques were employed to improve the stability and performance of the agent.

- Replay buffer:
  The trajectories of each agent are stored in a replay buffer which is sampled in the learning step. This breaks the
  correlation across the trajectories in each mini-batch and stabilizes the learning. The replay buffer size chosen is
  1e6, and the mini-batch size is 64.

- Soft-update of the target networks:
  Target networks were used for both the Actor and the Critic neural networks. The target networks offer consistent
  targets during for temporal differences. This method greatly stabilizes the learning. The target networks are updated
  after every learning step in a soft manner. The soft-update hyper-parameter 'tau' is 1e-3.

```python
target_weights = tau * current_weights + (1 - tau) * target_weights 
```

- Batch normalization:
  The input to each layer of the `feature` network (described below) as well as its output are all normalized using
  batch normalization. Each mini-batch is normalized such that the sample mean is 0 and variance is 1. Batch
  normalization improved the speed at which the agent solved the environment.

- Ornstein–Uhlenbeck (OU) noise to explore:
  The actions of the agent are noised using an Ornstein-Uhlenbeck simulated throughout the learning. This creates a
  small momentum in the exploration (the agent explores the action space in the same direction for a few time-steps) of
  the action space. The mean 'mu' of the OU process chosen is 0. The mean-reversion speed 'theta' was set to 0.15, the
  noise of the process 'sigma' was set to 0.2. The process was simulated using the Euler-Maruyama method.

- Multiple learning steps per time-step:
  Instead of learning on a single mini-batch sampled from the replay buffer per time-step, we executed two updates of
  the Actor-Critic network per time-step. This reduced the number of episodes required to solve the environment (but at
  a cost in terms of computation required per time-step).

For the Stochastic Gradient Descent, two 'Adam' optimizers are used, one for the critic update and one actor policy, the
learning rate is 1e-3 and the other parameters of the optimizers are kept to the default values. Note that the optimizer
are not updating the same set of parameters.

Other hyper-parameters include:

- The discount factor gamma is 0.99.
- The maximum number of steps per episode is 1001.

## Model Architecture

As mentioned in the 'Learning Algorithm' section above. A DDPG algorithm is used. This means that the neural network
used to solve the environment contains 3 blocks:

- The 'feature map' block: this first, 1-layer-deep block with ReLu activation, transforms the input state into features
  used in by the subsequent blocks. The input state as well as the output of the layer are normalized using batch
  normalization. The next two blocks are separated but share this block's outputs as inputs. Each layer contains 5 times
  the size of the observation space in cells with ReLu activations.

- The 'actor' block: this block is composed of a dense layer of the same size as the feature space with ReLu activation.
  Followed by a dense layer of the size of the action space with Tanh activation to ensure the policy stays within the
  bounds of the action space.

- The 'critic' block: the action is concatenated with the features and then used as inputs to a ReLu dense layer (still
  following the 5 times size rule). The block terminates with a single unit linear layer representing the (action,space)
  value.

## Results

The environment is solved after 109 episodes:

![image](https://user-images.githubusercontent.com/5805228/121542424-0a817880-ca00-11eb-99cc-96c821b436fb.png)

## Ideas for Future Work

The Trust Region Policy Optimization algorithm (TRPO), Truncated Natural Policy Gradient (TNPG) and Proximal Policy
Approximation (PPO) appear to be better suited than DDPG to solve this environment, it would be interesting to implement
them and compare them to this implementation. Additionally, it is quite compelling to try to distribute the training
across multiple hosts in the cloud. 
