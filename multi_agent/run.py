import supersuit as ss
from pettingzoo.mpe import simple_adversary_v2

# Creating simple_adversary_v2 parallel env
env = simple_adversary_v2.parallel_env()
action_spaces = env.action_spaces
env = ss.pad_observations_v0(env)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class="gym")

print(env.shared_act.shape)
