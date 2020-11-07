import numpy as np
from bandit_class import arm_functions
from scipy.stats import rv_discrete
from math import log
import sys


# User Non cmd line constants
file = "../instances/i-4.txt"
# Algorithms to be simulated
algos = ['uni-c-ucb']
# eps factor for deciding confidence set
eps = 0.11
# Horizon/ max number of iterations
horizon = 200001
STEP = 1000
# Upper bound on permitted rewards
B = 3
# TOL to prevent nans
TOL = 0.001

if __name__ == '__main__':
    for al in algos:
        for rs in range(50):
            # Read in file containing MAB instance information
            file_in = open(file, 'r')
            # Read in probab mass for above support - line 1
            dist_str = file_in.readline()
            # Convert string to float array
            dist = [float(x) for x in dist_str.split(' ')]
            # Convert to array
            dist = np.array(dist)
            # Generate support points as np array
            support = np.arange(len(dist))
            # Init the discrete latent random variable X
            # Create a custm discrete rv class
            custm = rv_discrete(name='custm', values=(support, dist))
            # Create instance of above
            X_rv = custm()
            # pre determining the values of X for given horizon and random seed
            X_realize = X_rv.rvs(horizon, random_state=rs)
            # Read in lines corresponding to arm functions
            functions_str = file_in.readlines()
            # Count number of arms
            n_arms = len(functions_str)
            # Convert strings to lists
            arm_list = []
            for i in range(n_arms):
                arm_list.append([float(s) for s in functions_str[i].split(' ')])
            # Covert to numpy array for sliced indexing
            arm_list = np.array(arm_list)
            # Compute expectations using matrix product
            expectations = np.dot(dist, np.transpose(arm_list))
            # Get maximum expected reward among all arms (optimal arm)
            exp_max = np.max(expectations)
            # Get index of optimal arm
            k_opt = np.argmax(expectations)
            # print(expectations)
            # print(k_opt)
            bandit_instance = arm_functions(arm_list)
            # Initialise cumulative reward
            REW = 0
            REG = 0
            # UCB: Vanilla Upper Confidence Bound Sampling algorithm
            if al == 'ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Initialize UCB index with infinity so that each arm sampled once at start
                I_ucb = np.repeat(np.inf, n_arms)
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin UCB based decisions
                for t in range(1, horizon):
                    # Determine arm to be sampled in current step, Ties are broken by index preference
                    k = np.argmax(I_ucb)
                    # Get reward based on arm choice
                    r = bandit_instance.sample(k, X_realize[t-1])
                    # Update cummulative reward
                    REW = REW + r
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update ucb index value
                    I_ucb = mu_hat + B * np.sqrt(2 * log(t) / nsamps)
                    REG = REG + arm_list[k_opt][X_realize[t-1]] - r
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}\n".format(al, rs, eps, t, REG, exp_max * t - REW))

            # UCB: C - Upper Confidence Bound Sampling algorithm
            # As described in main reference paper
            elif al == 'c-ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Square matrix Array to hold empirical pseudo rewards
                phi_hat = np.zeros((n_arms, n_arms))
                # Array to hold pseudo gap values
                del_hat = np.zeros(n_arms)
                # Initilaise UCB index with infinity so that each arm sampled once at start
                I_ucb = np.repeat(np.inf, n_arms)
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin C - UCB based decisions
                for t in range(1, horizon):
                    # Determine the arm that has been pulled the most number of times uptill
                    # iteration t - 1
                    k_max = np.argmax(nsamps)
                    # Get pseudo gaps wrt arm k_max, second term on RHS is a vector
                    del_hat = mu_hat[k_max] - phi_hat[k_max]
                    # Identify arms competitive wrt arm k_max
                    is_comp = del_hat <= 0
                    # Determine arm to be sampled in current step, Ties are broken by index preference
                    max = 0
                    k = 0
                    for i in range(n_arms):
                        if I_ucb[i] > max and is_comp[i]:
                            k = i   # Update arm
                            max = I_ucb[i]
                    # Get reward based on arm choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update cummulative reward
                    REW = REW + r
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update ucb index value
                    I_ucb = np.float64(mu_hat + B * np.sqrt(2 * log(t+TOL) / nsamps))
                    # Compute and update pseudo rewards
                    phi_hat[k] = ((nsamps[k] - 1) * phi_hat[k] + bandit_instance.get_pseudo_rew(k, r)) / nsamps[k]
                    REG = REG + arm_list[k_opt][X_realize[t-1]] - r
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}\n".format(al, rs, eps, t, REG, exp_max * t - REW))

            # UCB: Uniform-C - Upper Confidence Bound Sampling algorithm
            # As described in our report
            elif al == 'uni-c-ucb':
                # Array to hold estimated pseudo distribution of X, uniform prior
                dist_hat = np.repeat(1/len(support), len(support))
                # dist_hat = np.repeat(TOL, len(support))
                # List to hold confidence set, initialise with complete support
                Cstar = list(range(n_arms))
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Initilaise UCB index with infinity so that each arm sampled once at start
                I_ucb = np.repeat(np.inf, n_arms)
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin C - UCB based decisions
                for t in range(1, horizon):
                    # Init list for competitive arms, initially assume all compettive
                    comp_arms = list(range(n_arms))
                    # Get current empirical estimates
                    pseudo_exp = np.dot(dist_hat, np.transpose(arm_list))
                    # determine the competitive set with respect to Cstar
                    for i in range(n_arms):
                        for j in range(n_arms):
                            # If the ith arm func is less the j th arm func for all x in Cstar
                            if np.all(arm_list[i][Cstar] < arm_list[j][Cstar]) and pseudo_exp[i] < pseudo_exp[j] and i in comp_arms:
                                # Remove from competitive set
                                comp_arms.remove(i)
                    # print(comp_arms)
                    # Determine arm to be sampled in current step, Ties are broken by index preference
                    max_ucb = 0
                    k = 0
                    for i in comp_arms:
                        if I_ucb[i] > max_ucb:
                            k = i  # Update arm
                            max_ucb = I_ucb[i]
                    # Get reward based on arm choice
                    r = arm_list[k][X_realize[t - 1]]
                    # Update cummulative reward
                    REW = REW + r
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update ucb index value
                    I_ucb = np.float64(mu_hat + B * np.sqrt(2 * log(t + TOL) / nsamps))
                    # Construct an array with indices as 1 for points that lie in the inverse
                    incr_indices = bandit_instance.get_inverse(k, r).astype(int)
                    # Get change vector by taking dot pruct with prior distribution
                    # WARNING: This biasing of the update is not recommneded and
                    # has been corrected in a lter version
                    incr_vector = incr_indices*dist_hat
                    # Update the pseudo distribution
                    dist_hat = (t*dist_hat + incr_vector/np.sum(incr_vector))/(t+1)
                    # Get indices that would sort distribution array
                    dist_sort = np.argsort(dist_hat)
                    # Reverse for descending
                    dist_sort = dist_sort[::-1]
                    # Update eps confidence set
                    Cstar = []
                    sum_prob = 0
                    ctr = 0
                    while sum_prob < 1 - eps :
                        sum_prob = sum_prob + dist_hat[dist_sort[ctr]]
                        Cstar.append(dist_sort[ctr])
                        ctr = ctr + 1
                    REG = REG + arm_list[k_opt][X_realize[t-1]] - r
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}\n".format(al, rs, eps, t, REG, exp_max * t - REW))
            # UCB: Uniform-C - Upper Confidence Bound Sampling algorithm
            # As described in our report
            elif al == 'uni-c-ucb2':
                # Array to hold estimated pseudo distribution of X, uniform prior
                dist_hat = np.repeat(1/len(support), len(support))
                # List to hold confidence set, initialise with complete support
                Cstar = list(range(n_arms))
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Initilaise UCB index with infinity so that each arm sampled once at start
                I_ucb = np.repeat(np.inf, n_arms)
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin C - UCB based decisions
                for t in range(1, horizon):
                    # Init list for competitive arms, initially assume all compettive
                    comp_arms = list(range(n_arms))
                    # Get current empirical estimates
                    pseudo_exp = np.dot(dist_hat, np.transpose(arm_list))
                    # determine the competitive set with respect to Cstar
                    for i in range(n_arms):
                        for j in range(n_arms):
                            # If the ith arm func is less the j th arm func for all x in Cstar
                            if np.all(arm_list[i][Cstar] < arm_list[j][Cstar]) and pseudo_exp[i] < pseudo_exp[j] and i in comp_arms:
                                # Remove from competitive set
                                comp_arms.remove(i)
                    # print(comp_arms)
                    # Determine arm to be sampled in current step, Ties are broken by index preference
                    max_ucb = 0
                    k = 0
                    for i in comp_arms:
                        if I_ucb[i] > max_ucb:
                            k = i  # Update arm
                            max_ucb = I_ucb[i]
                    # Get reward based on arm choice
                    r = arm_list[k][X_realize[t - 1]]
                    # Update cummulative reward
                    REW = REW + r
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update ucb index value
                    I_ucb = np.float64(mu_hat + B * np.sqrt(2 * log(t + TOL) / nsamps))
                    # Construct an array with indices as 1 for points that lie in the inverse
                    incr_indices = bandit_instance.get_inverse(k, r).astype(int)
                    # Update the pseudo distribution as described in Juneja et al.
                    dist_hat = (t*dist_hat + incr_indices/np.sum(incr_indices))/(t + 1)
                    # Get indices that would sort distribution array
                    dist_sort = np.argsort(dist_hat)
                    # Reverse for descending
                    dist_sort = dist_sort[::-1]
                    # Update eps confidence set
                    Cstar = []
                    sum_prob = 0
                    ctr = 0
                    while sum_prob < 1 - eps :
                        sum_prob = sum_prob + dist_hat[dist_sort[ctr]]
                        Cstar.append(dist_sort[ctr])
                        ctr = ctr + 1
                    REG = REG + arm_list[k_opt][X_realize[t-1]] - r
                    if t % STEP == 0:
                        print(dist_hat)
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}\n".format(al, rs, eps, t, REG, exp_max * t - REW))
            # No valid algorithm selected
            else:
                print("Invalid algorithm selected")
                # Don't print REG
                exit(-1)
