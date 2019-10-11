import gym
import numpy as np

env = gym.make("MountainCar-v0")
#env.reset()

# print some info about the env
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
print(f"goal position: {env.goal_position}")

LEARNING_RATE = 0.2
DISCOUNT = 0.95
EPISODES = 5000
SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/ DISCRETE_OS_SIZE

print(discrete_os_win_size)

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# The low and high is based on how the env returns reward, by default the reward is -1
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

# print(q_table)
# print(q_table.shape)

for episode in range(EPISODES):

    discrete_state = get_discrete_state(env.reset())

    if episode % SHOW_EVERY == 0:
        print(f"episode: {episode}")
        render = True
    else:
        render = False
    

    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, ) ]
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT*max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()


