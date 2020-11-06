# UCB: Uniform-C - Upper Confidence Bound Sampling algorithm
    # As described in our report
    elif al == 'exp-c-ucb':
        # Array to hold estimated pseudo distribution of X, uniform prior
        dist_hat = np.repeat(1 / len(support), len(support))
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
            # determine the competitive set with respect to Cstar
            for i in range(n_arms):
                for j in range(n_arms):
                    # Compute expected reward of arm i given x in Cstar
                    expecI = 0
                    expecJ = 0
                    for x in Cstar:
                        expecI = expecI + dist_hat[x]*arm_list[i][x]
                        expecJ = expecJ + dist_hat[x]*arm_list[j][x]
                    # If the ith arm func is less the j th arm func for all x in Cstar
                    if expecI < expecJ and i in comp_arms:
                        # Remove from competitive set
                        comp_arms.remove(i)
            if len(comp_arms) > 2:
                print("Found")
            # Determine arm to be sampled in current step, Ties are broken by index preference
            max = 0
            for i in comp_arms:
                if I_ucb[i] > max:
                    k = i  # Update arm
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
            I_ucb = np.float64(mu_hat + B * np.sqrt(2 * log(t + TOL) / nsamps))
            # Construct an array with indices as 1 for points that lie in the inverse
            incr_indices = bandit_instance.get_inverse(k, r).astype(int)
            # Get change vector by taking dot pruct with prior distribution
            incr_vector = incr_indices * dist_hat
            # Update the pseudo distribution
            dist_hat = (t * dist_hat + incr_vector / np.sum(incr_vector)) / (t + 1)
            # Get indices that would sort distribution array
            dist_sort = np.argsort(dist_hat)
            # Update eps confidence set
            Cstar = []
            sum_prob = 0
            ctr = 0
            while sum_prob < 1 - eps:
                sum_prob = sum_prob + dist_hat[dist_sort[ctr]]
                Cstar.append(dist_sort[ctr])
                ctr = ctr + 1
            # print(Cstar)
            if t in sample_horizons:
                sys.stdout.write(
                    "{0}, {1}, {2}, {3}, {4}, {5:.2f}\n".format(file, al, rs, eps, t, exp_max * t - REW))

# UCB: Uniform-C - Upper Confidence Bound Sampling algorithm
# As described in our report
elif al == 'uni-c-ucb2':
dist_hat = np.repeat(1 / len(support), len(support))
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
    # Get arm index sampled most times
    k_max = np.argmax(nsamps)
    # Init list for competitive arms, initially assume all compettive
    comp_arms = list(range(n_arms))
    # determine the competitive set with respect to Cstar
    for i in range(n_arms):
        # If the ith arm func is less the j th arm func for all x in Cstar
        if np.all(arm_list[i][Cstar] < arm_list[k_max][Cstar]) and i in comp_arms:
            # Remove from competitive set
            comp_arms.remove(i)
    # Determine arm to be sampled in current step, Ties are broken by index preference
    max = 0
    for i in comp_arms:
        if I_ucb[i] > max:
            k = i  # Update arm
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
    I_ucb = np.float64(mu_hat + B * np.sqrt(2 * log(t + TOL) / nsamps))
    # Construct an array with indices as 1 for points that lie in the inverse
    incr_indices = bandit_instance.get_inverse(k, r).astype(int)
    # Get change vector by taking dot pruct with prior distribution
    incr_vector = incr_indices * dist_hat
    # Update the pseudo distribution
    dist_hat = (t * dist_hat + incr_vector / np.sum(incr_vector)) / (t + 1)
    # Get indices that would sort distribution array
    dist_sort = np.argsort(dist_hat)
    # Reverse for descending
    dist_sort = dist_sort[::-1]
    # Update eps confidence set
    Cstar = []
    sum_prob = 0
    ctr = 0
    while sum_prob < 1 - eps:
        sum_prob = sum_prob + dist_hat[dist_sort[ctr]]
        Cstar.append(dist_sort[ctr])
        ctr = ctr + 1
    REG = REG + arm_list[k_opt][X_realize[t - 1]] - r
    if t % 100 == 0:
        sys.stdout.write(
            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}\n".format(al, rs, eps, t, REG, exp_max * t - REW))