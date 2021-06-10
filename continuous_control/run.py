from unityagents import UnityEnvironment

from utils import run
from agent import DDPGAgent

env = UnityEnvironment(file_name="Reacher_Linux_NoVis/Reacher.x86_64", base_port=50001)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print("Number of agents:", num_agents)

# size of each action
action_size = brain.vector_action_space_size
print("Size of each action:", action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(
    f"There are {states.shape[0]} agents. Each observes a state with length: {state_size}"
)
print("The state for the first agent looks like:", states[0])

agent = DDPGAgent(state_size, action_size, seed=42)


run(env, agent, train_mode=True)

env.close()
