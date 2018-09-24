import numpy as np
import random
import math
import matplotlib.pyplot as plt

######################################################
# Tejas Jha
# 24 September 2018
# EECS 498 Special Topics: Reinforcement Learning
# Homework 1
# Code corresponding to Q4 and Q5
######################################################


# For Q4:
#
# This program provides experimentations with Epsilon-Greedy, 
# Optimistic Initial Values, and UCB algorithms to learn solutions
# to a 5-arm bandit problem.
#
# Consider the arm rewards to be Bernoulli Distribution with means 
# [0.1, 0.275, 0.45, 0.625, 0.8]
#
# Each algorithm should be performed for 2000 runs for each parameter set
# Each run should choose actions over 1000 time steps

# For Q5
#
#
#
#


# Creation of class to store Bernoulli distribution mean and return reward
class Arm:
    # Initialize arm given 
    def __init__(self, mean):
        self.mean = mean
    
    # Simulate Bernoulli distribution
    # Get random float between 0 and 1, return 0 if greater than mean, 1 otherwise
    def reward(self):
        # The code below will perform the same general functionality as using np.random.binomial
        # However, it is slightly faster in performace, so I use it here instead
        
        if random.random() > self.mean:
            return 0.0
        else:
            return 1.0

        # This can be used instead of the above code for a more library method of 
        # getting a bernoulli distribution, but it is not necessarily needed
        # return np.random.binomial(1, self.mean, 1)

    # Return the mean of the current Arm's Bernoulli distribution
    def getMean(self):
        return self.mean
    


# Creation of a class to store details during each time step and house
# necessary functions needed during usage.
# Note: I used the code found here: https://gist.github.com/nagataka/c02b9acd6e8a8d7696e09f8a129d3362
#       for inspiration and guidance on how to construct my own functions and organize my code
class EpsilonGreedy:
    # Initialize class values
    # epsilon - the rate of exploration (randomly choosing an arm)
    # counts - N(a), number of pulls for each arm
    # values - Q(a) average reward received from each arm (action-value estimates)
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        self.regret = [ ]
        self.rewards = [ ]
        self.optimal_count = 0
        self.optimal = [ ]

    # Performs a single time step update through epsilon-greedy algorithm
    def step(self, iterations, arms):
        # Step 1 - Determine which arm to pick next
        selected_arm_idx = 0
        if random.random() > self.epsilon:
            # EXPLOITATION
            # Choose action with largest expected action-value (break ties randomly)
            max_value = max(self.values)
            all_max_idx = [idx for idx, val in enumerate(self.values) if val == max_value]
            selected_arm_idx = all_max_idx[0]

            # (Break ties randomly)
            if len(all_max_idx) > 1:
                # Randomly choose an index from all_max_idx
                selected_arm_idx = random.choice(all_max_idx)
        else:
            # EXPLORATION
            # Choose random index to explore
            selected_arm_idx = random.choice(range(5))

        # Step 2 - Get the reward for that arm
        reward = arms[selected_arm_idx].reward()
        # Step 3 - Update count
        self.counts[selected_arm_idx] += 1
        # Step 4 - Update action-value
        self.values[selected_arm_idx] += (1 / self.counts[selected_arm_idx])*(reward - self.values[selected_arm_idx])

        # Update reward at time step
        self.rewards.append(reward)

        # Update cummulative regret value for this time step
        self.regret.append(iterations*(0.8) - sum(self.rewards))

        # If optimal choice was chosen, update optimal_count
        if selected_arm_idx == 4:
            self.optimal_count += 1
        # Update Optimal Percentages
        self.optimal.append(self.optimal_count/iterations)


    # Get the action-value estimates
    def getValues(self):
        return self.values

    # Get the Cumulative regret at certain moment in time
    def getRegret(self):
        return self.regret

    # Get the rewards at certain moments in time
    def getRewards(self):
        return self.rewards
    
    # Get the total number of times the optimal arm was chosen
    def getOptimal(self):
        return self.optimal

# Creation of another Class variable that may be fixed to be universal type
class ModelClass:
    def __init__(self, epsilon=0, counts=np.zeros(5), values=np.zeros(5), c=0):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        self.c = c
        self.regret = [ ]
        self.rewards = [ ]
        self.optimal_count = 0
        self.optimal = [ ]

    def step(self, iterations, arms):

        # Step 1 -Select the Action with the largest upper bound
        # If count is 0, the index is given priority
        all_zero_idx = [idx for idx, val in enumerate(self.counts) if val == 0]
        selected_arm_idx = 0
        #updated_val = 0
        if len(all_zero_idx) > 0:
            selected_arm_idx = random.choice(all_zero_idx)
        else:
            # Select the arm with the largest estimated upper bound
            estimates = self.values.copy()

            for idx in range(len(estimates)):
                estimates[idx] += self.c*math.sqrt(math.log(iterations)/self.counts[idx])
            
            max_value = max(estimates)
            all_max_idx = [idx for idx, val in enumerate(estimates) if val == max_value]
            selected_arm_idx = all_max_idx[0]

            # (Break ties randomly)
            if len(all_max_idx) > 1:
                # Randomly choose an index from all_max_idx
                selected_arm_idx = random.choice(all_max_idx)
            
            #updated_val = estimates[selected_arm_idx]
        
        # Step 2 - Get the reward for that arm
        reward = arms[selected_arm_idx].reward()
        # Step 3 - Update count
        self.counts[selected_arm_idx] += 1
        # Step 4 - Update action-value
        self.values[selected_arm_idx] += (1 / self.counts[selected_arm_idx])*(reward - self.values[selected_arm_idx])
        
        # Update reward at time step
        self.rewards.append(reward)

        # Update cummulative regret value for this time step
        self.regret.append(iterations*(0.8) - sum(self.rewards))

        # If optimal choice was chosen, update optimal_count
        if selected_arm_idx == 4:
            self.optimal_count += 1
        # Update Optimal Percentages
        self.optimal.append(self.optimal_count/iterations)

    # Get the action-value estimates
    def getValues(self):
        return self.values

    # Get the Cumulative regret at certain moment in time
    def getRegret(self):
        return self.regret

    # Get the rewards at certain moments in time
    def getRewards(self):
        return self.rewards
    
    # Get the total number of times the optimal arm was chosen
    def getOptimal(self):
        return self.optimal



# Epsilon-Greedy Algorithm run 2000 times given epsilon value
def epsilon_greedy_alg(epsilon, arms):

    '''
    # Get optimal arm index
    optimal_arm_idx = 0
    optimal_mean = 0
    i = 0
    for arm in arms:
        if arm.getMean > optimal_mean:
            optimal_arm_idx = i
        i += 1
    '''

    # Print message to output describing Algorithm being run
    print("**********************************************************************")
    print("Performing Epsilon-Greedy Algorithm 2000 times with epsilon = " + str(epsilon))

    # Store sum of regrets
    regrets_avg = np.zeros([1,1000])
    # Store average rewards at each time step for all runs
    avg_rewards = np.zeros([1,1000])
    # Average the % Optimal Action at each time step for all runs
    percent_optimal_action = np.zeros([1,1000])

    # Loop through algorithm for 2000 runs
    for current_run in range(2000):

        # Each run should set a distinct random seed
        random.seed(current_run + 1)
        # Create EpsilonGreedy data type initialized with epsilon and 0 for counts and values
        model = EpsilonGreedy(epsilon, np.zeros(5), np.zeros(5))

        # Perform a single run through algorithm using 1000 time steps
        for i in range(1000):
            model.step(i+1, arms)

        # Update sum of regrets
        np_arr = np.array(model.getRegret())
        np_arr = np_arr.reshape(1,1000)
        np_arr = np_arr/2000.0
        regrets_avg += np_arr

        # Update average of rewards for all timesteps
        avg_rewards += (np.array(model.getRewards()).reshape(1,1000))

        # Update averages of % Optimal Action at each time step
        percent_optimal_action += (np.array(model.getOptimal()).reshape(1,1000))

        if (current_run + 1) % 500 == 0:
            print("     Completed Run #" + str(current_run + 1))
            print(model.getValues())

    avg_rewards = avg_rewards / 2000.0
    percent_optimal_action = percent_optimal_action / 2000.0

    print("Completed Runs for epsilon = " + str(epsilon))
    #print("regrets_avg = " + str(regrets_avg))
    print("**********************************************************************")
    return regrets_avg, avg_rewards, percent_optimal_action


# Optimistic Initial Value Algorithm run 2000 times given inital value (assume greedy arm selection)
def optimistic_initial_value_alg(initial_val, arms):

    # Print message to output describing Algorithm being run
    print("**********************************************************************")
    print("Performing Optimistic Initial Value Algorithm 2000 times with inital value = " + str(initial_val))

    # Store sum of regrets
    regrets_avg = np.zeros([1,1000])
    # Store average rewards at each time step for all runs
    avg_rewards = np.zeros([1,1000])
    # Average the % Optimal Action at each time step for all runs
    percent_optimal_action = np.zeros([1,1000])

    # Loop through algorithm for 2000 runs
    for current_run in range(2000):

        # Each run should set a distinct random seed
        random.seed(current_run + 1)
        # Create EpsilonGreedy data type initialized with epsilon = 0 and 0 for counts and inital values
        initialized_values_array = np.full(5,initial_val)
        model = EpsilonGreedy(0, np.zeros(5), initialized_values_array)

        # Perform a single run through algorithm using 1000 time steps
        for i in range(1000):
            model.step(i+1,arms)
        
        # Update sum of regrets
        np_arr = np.array(model.getRegret())
        np_arr = np_arr.reshape(1,1000)
        np_arr = np_arr/2000.0
        regrets_avg += np_arr

        # Update average of rewards for all timesteps
        avg_rewards += (np.array(model.getRewards()).reshape(1,1000))

        # Update averages of % Optimal Action at each time step
        percent_optimal_action += (np.array(model.getOptimal()).reshape(1,1000))

        if (current_run + 1) % 500 == 0:
            print("     Completed Run #" + str(current_run + 1))
            print(model.getValues())

    avg_rewards = avg_rewards / 2000.0
    percent_optimal_action = percent_optimal_action / 2000.0

    print("Completed Runs for initial value = " + str(initial_val))
    print("**********************************************************************")
    return regrets_avg, avg_rewards, percent_optimal_action


# Upper Confidence Bound (UCB) Algorithm run 2000 times with given c parameter
def upper_confidence_bound_alg(c, arms):

    # Print message to output describing Algorithm being run
    print("**********************************************************************")
    print("Performing UCB Algorithm 2000 times with c = " + str(c))

    # Store sum of regrets
    regrets_avg = np.zeros([1,1000])
    # Store average rewards at each time step for all runs
    avg_rewards = np.zeros([1,1000])
    # Average the % Optimal Action at each time step for all runs
    percent_optimal_action = np.zeros([1,1000])

    # Loop through algorithm for 2000 runs
    for current_run in range(2000):

        # Each run should set a distinct random seed
        random.seed(current_run + 1)
        
        # Initialize model with c value
        model = ModelClass(0,np.zeros(5),np.zeros(5),c)

        # Perform a single run through algorithm using 1000 time steps
        for i in range(1000):
            model.step(i+1,arms)

        # Update sum of regrets
        np_arr = np.array(model.getRegret())
        np_arr = np_arr.reshape(1,1000)
        np_arr = np_arr/2000.0
        regrets_avg += np_arr

        # Update average of rewards for all timesteps
        avg_rewards += (np.array(model.getRewards()).reshape(1,1000))

        # Update averages of % Optimal Action at each time step
        percent_optimal_action += (np.array(model.getOptimal()).reshape(1,1000))

        if (current_run + 1) % 500 == 0:
            print("     Completed Run #" + str(current_run + 1))
            print(model.getValues())

    avg_rewards = avg_rewards / 2000.0
    percent_optimal_action = percent_optimal_action / 2000.0

    print("Completed Runs for c = " + str(c))
    print("**********************************************************************")
    return regrets_avg, avg_rewards, percent_optimal_action


# Question 5 Functions

# Helper function to assist inintialization of P
def fill(P, s, neg, zero):
    for elem in neg:
        P[s][elem].append([0.8, s, -1.0])
        P[s][elem].append([0.2, s, 0.0])
    for elem in zero:
        if elem == 0:
            P[s][elem].append([0.8, s - 5, 0.0])
        elif elem == 1:
            P[s][elem].append([0.8, s + 1, 0.0])
        elif elem == 2:
            P[s][elem].append([0.8, s + 5, 0.0])
        elif elem == 3:
            P[s][elem].append([0.8, s - 1, 0.0])
        else: 
            print("Error: action was entered larger than 3!")
        P[s][elem].append([0.2, s, 0.0])

# Function to work out the transition probability and rewards for the MDP described in the problem statement
def gridworld(slip_prob = 0.2):
    # slip_prob is the probability that the agent slips

    # Initialize P organization
    P = {}
    for s in range(25):
        P[s] = {0: [], 1: [], 2: [], 3: []}

    # Initialize P values
    for s in range(25):
        if s == 1:
            for a in range(4):
                P[s][a].append([1.0, 21, 10.0])
        elif s == 3:
            for a in range(4):
                P[s][a].append([1.0, 13, 5.0])
        else:
            zero = [0, 1, 2 ,3]
            neg = [ ]

            if s < 5:
                zero.remove(0)
                neg.append(0)

            if s % 5 == 4:
                zero.remove(1)
                neg.append(1)

            if s > 19:
                zero.remove(2)
                neg.append(2)
            
            if s % 5 ==0:
                zero.remove(3)
                neg.append(3)
            
            fill(P, s, neg, zero)

    return P

# Policy class assumes map is square
class Policy:

    def __init__(self, distributions=np.full([25,4],0)):
        self.distributions = distributions
    
    def getDistributions(self):
        return self.distributions

    def setDistributions(self, distributions):
        self.distributions = distributions

    def print(self):
        dim = int(math.sqrt(self.distributions.shape[0]))
        for y in range(dim):
            output_str = ""
            for x in range(dim):
                max_value = max(self.distributions[5*y+x])
                all_max_idx = [idx for idx, val in enumerate(self.distributions[5*y+x]) if val == max_value]
                for idx in all_max_idx:
                    if idx == 0:
                        output_str += "n"
                    elif idx == 1:
                        output_str += "e"
                    elif idx == 2:
                        output_str += "s"
                    elif idx == 3:
                        output_str += "w"
                    else:
                        print("Error: wrong idx in printing Policy")
                output_str += " "
            print(output_str)

    def get_state_action(self, state):
        #max_prob = max(self.distributions[state])
        #all_max_idx = [idx for idx, val in enumerate(self.distributions[state]) if val == max_value]
        all_prob = self.distributions[state]
        expectation = 0
        for i in range(4):
            expectation += i*all_prob[i]
        return expectation

    def set_state_action(self, state, d):
        for key, value in d.items():
            self.distributions[state][key] = value

class uniform_policy(Policy):

    def __init__(self):
        Policy.__init__(self, distributions=np.full([25,4],0.25))


# Helper function to calculate updated V(s)
def getSum(V, P, policy, gamma, s):
    external_sum = 0
    for a in range(4):
        internal_sum = 0
        for entry in range(len(P[s][a])):
            prob = P[s][a][entry][0]
            next_state = P[s][a][entry][1]
            reward = P[s][a][entry][2]
            internal_sum += prob * (reward + gamma * V[next_state//5][next_state%5])
        external_sum += policy.getDistributions()[s][a] * internal_sum
    return external_sum

# Helper function to get dictionary of updated probabilities
def getProb(P, policy, gamma, s, V):
    best_actions = []
    best_estimate = -10
    for a in range(4):
        estimate = 0
        for entry in range(len(P[s][a])):
            prob = P[s][a][entry][0]
            next_state = P[s][a][entry][1]
            reward = P[s][a][entry][2]
            estimate += prob * (reward + gamma * V[next_state//5][next_state%5])
        if (estimate - best_estimate) > 0.01:
            best_actions = [a]
            best_estimate = estimate
        elif abs(estimate - best_estimate) < 0.01:
            # Note: Comparing floats, so leaving precision to 2 decimal places
            best_actions.append(a)
            
    updated_distributions = { }
    if len(best_actions) == 1:
        for a in best_actions:
            updated_distributions[a] = 1
        for a in range(4):
            if a != best_actions[0]:
                updated_distributions[a] = 0
    if len(best_actions) == 2:
        for a in best_actions:
            updated_distributions[a] = 0.5
        for a in range(4):
            if a != best_actions[0] and a != best_actions[1]:
                updated_distributions[a] = 0

    # Nothing said about ties between 3 or 4 actions...
    
    return updated_distributions


# Policy Evaluation Algorithm
def policy_eval(P, policy=uniform_policy(), theta=0.0001, gamma=0.9):
    V = np.zeros([5,5])
    delta = 1
    while delta >= theta:
        delta = 0
        for s in range(25):
            v = V[s//5][s%5]
            V[s//5][s%5] = getSum(V,P,policy,gamma,s)
            delta = max(delta, abs(v - V[s//5][s%5]))
    return V

# Policy Iteration Algorithm
def policy_iter(P, theta=0.0001, gamma=0.9):
    policy_stable = False
    policy = uniform_policy()
    while policy_stable == False:
        policy_stable = True
        for s in range(25):
            old_action = policy.get_state_action(s)
            V = policy_eval(P, policy=policy, theta=theta, gamma=gamma)
            policy.set_state_action(s,getProb(P,policy,gamma,s,V))
            if old_action != policy.get_state_action(s):
                policy_stable = False
    V = policy_eval(P, policy=policy, theta=theta, gamma=gamma)
    return V, policy

def main():

    # Question 4 Code

    # Create List of arm class variables called arms set with means for Bernoulli distributions
    means = [0.1, 0.275, 0.45, 0.625, 0.8]
    arms = np.array(list(map(Arm, means)))

    fig = plt.figure(1)
    fig.suptitle('Graphs for Q4 By Tejas Jha', fontsize=16)
    times = (np.array(range(1,1001)).reshape(1,1000))
    regrets_avg = np.zeros([3,1000])
    avg_rewards = np.zeros([3,1000])
    percent_optimal_action = np.zeros([3,1000])

    # Perform Epsilon-Greedy algorithm with Q1 = 0 and 
    # for each epsilon = [0.01, 0.1, 0.3]
    epsilon_list = [0.01, 0.1, 0.3]
    for i in range(len(epsilon_list)):
        #regrets_avg[i], avg_rewards[i], percent_optimal_action[i] = epsilon_greedy_alg(epsilon_list[i], arms)
        print("Finished with Epsilon-Greedy Algorithm for epsilon = " + str(epsilon_list[i]))

    plt.subplot(331)
    #plt.title("Epsilon-Greedy Cumulative Regret vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.plot(times[0], regrets_avg[0], 'r', label='Epsilon = 0.01')
    plt.plot(times[0], regrets_avg[1], 'b', label='Epsilon = 0.1')
    plt.plot(times[0], regrets_avg[2], 'g', label='Epsilon = 0.3')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.subplot(332)
    #plt.title("Epsilon-Greedy Averaged Reward vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Averaged Reward")
    plt.plot(times[0], avg_rewards[0], 'r', label='Epsilon = 0.01')
    plt.plot(times[0], avg_rewards[1], 'b', label='Epsilon = 0.1')
    plt.plot(times[0], avg_rewards[2], 'g', label='Epsilon = 0.3')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.subplot(333)
    #plt.title("Epsilon-Greedy % Optimal Action vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("% Optimal Action")
    plt.plot(times[0], percent_optimal_action[0], 'r', label='Epsilon = 0.01')
    plt.plot(times[0], percent_optimal_action[1], 'b', label='Epsilon = 0.1')
    plt.plot(times[0], percent_optimal_action[2], 'g', label='Epsilon = 0.3')
    plt.legend(loc='upper left', fontsize='x-small')

    regrets_avg = np.zeros([3,1000])
    avg_rewards = np.zeros([3,1000])
    percent_optimal_action = np.zeros([3,1000])
    
    # Perform Optimistic Initial Value algorithm with epsilon = 0 (always greedy)
    # for each Q1 = [1,5,50]
    initial_val_list = [1.0, 5.0, 50.0]
    for i in range(len(initial_val_list)):
        #regrets_avg[i], avg_rewards[i], percent_optimal_action[i] = optimistic_initial_value_alg(initial_val_list[i],arms)
        print("Finished with Optimistic Initial Value Algorithm for initial value = " + str(initial_val_list[i]))

    plt.subplot(334)
    #plt.title("OIV Cumulative Regret vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.plot(times[0], regrets_avg[0], 'r', label='Q1 = 1.0')
    plt.plot(times[0], regrets_avg[1], 'b', label='Q1 = 5.0')
    plt.plot(times[0], regrets_avg[2], 'g', label='Q1 = 50.0')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.subplot(335)
    #plt.title("OIV Averaged Reward vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Averaged Reward")
    plt.plot(times[0], avg_rewards[0], 'r', label='Q1 = 1.0')
    plt.plot(times[0], avg_rewards[1], 'b', label='Q1 = 5.0')
    plt.plot(times[0], avg_rewards[2], 'g', label='Q1 = 50.0')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.subplot(336)
    #plt.title("OIV % Optimal Action vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("% Optimal Action")
    plt.plot(times[0], percent_optimal_action[0], 'r', label='Q1 = 1.0')
    plt.plot(times[0], percent_optimal_action[1], 'b', label='Q1 = 5.0')
    plt.plot(times[0], percent_optimal_action[2], 'g', label='Q1 = 50.0')
    plt.legend(loc='upper left', fontsize='x-small')

    regrets_avg = np.zeros([3,1000])
    avg_rewards = np.zeros([3,1000])
    percent_optimal_action = np.zeros([3,1000])

    # Perform UCB algorithm with Q1 = 0
    # for each c = [0.2, 1, 2]
    c_vals = [0.2, 1, 2]
    for i in range(len(c_vals)):
        #regrets_avg[i], avg_rewards[i], percent_optimal_action[i] = upper_confidence_bound_alg(c_vals[i], arms)
        print("Finished with UCB Algorithm for c = " + str(c_vals[i]))

    plt.subplot(337)
    #plt.title("UCB Cumulative Regret vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.plot(times[0], regrets_avg[0], 'r', label='c = 0.2')
    plt.plot(times[0], regrets_avg[1], 'b', label='c = 1.0')
    plt.plot(times[0], regrets_avg[2], 'g', label='c = 2.0')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.subplot(338)
    #plt.title("UCB Averaged Reward vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("Averaged Reward")
    plt.plot(times[0], avg_rewards[0], 'r', label='c = 0.2')
    plt.plot(times[0], avg_rewards[1], 'b', label='c = 1.0')
    plt.plot(times[0], avg_rewards[2], 'g', label='c = 2.0')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.subplot(339)
    #plt.title("UCB % Optimal Action vs Time Step")
    plt.xlabel("Time Step")
    plt.ylabel("% Optimal Action")
    plt.plot(times[0], percent_optimal_action[0], 'r', label='c = 0.2')
    plt.plot(times[0], percent_optimal_action[1], 'b', label='c = 1.0')
    plt.plot(times[0], percent_optimal_action[2], 'g', label='c = 2.0')
    plt.legend(loc='upper left', fontsize='x-small')

    plt.show()

    ###########################################################################################

    # Question 5 Code
    
    P = gridworld(slip_prob=0.2)
    V = policy_eval(P)
    np.set_printoptions(precision=2)
    print(V)

    V, policy = policy_iter(P)
    print(V)
    policy.print()



    



if __name__ == "__main__":
    main()
    