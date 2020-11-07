import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

x_points = np.arange(100, 200000, 100)
y_points = np.zeros_like(x_points)
LABELS = ['ucb', 'c-ucb', 'uni-c-cub']
in_file = '../results/i3-out-final-paper.txt'
bandit_data = pd.read_csv(in_file, sep=", ", header=None)
bandit_data.columns = ["algo", "rs", "eps", "horizon", "REG", "EXP_REG"]
plt.figure(figsize=(4, 3))

# Plot results for ucb
ucb = bandit_data.loc[bandit_data["algo"] == "ucb"]
c_ucb = bandit_data.loc[bandit_data["algo"] == "c-ucb"]
uni_ucb = bandit_data.loc[bandit_data["algo"] == "uni-c-ucb"]

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = ucb.loc[ucb["horizon"] == x_points[i]]["REG"].mean()
print(y_points)
plt.plot(x_points, y_points, color='r', linewidth=3)
#plt.plot(np.log10(x_points), y_points, color='b')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = c_ucb.loc[c_ucb["horizon"] == x_points[i]]["REG"].mean()
print(y_points)
plt.plot(x_points, y_points, color='b', linewidth=3)
#plt.plot(np.log10(x_points), y_points, color='r')

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = uni_ucb.loc[uni_ucb["horizon"] == x_points[i]]["REG"].mean()
print(y_points)
plt.plot(x_points, y_points, color='g', linewidth=3)
#plt.plot(np.log10(x_points), y_points, color='g')

plt.legend(LABELS)
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))
plt.xticks(np.arange(0, max(x_points)+1, 50000))
plt.xlabel("Number of rounds T", fontweight="bold")
plt.ylabel("Cumulative Regret", fontweight="bold")
plt.title("Instance 3: UCB vs. CUCB vs. U-CUCB", fontweight="bold")
plt.yticks()
plt.savefig(in_file + "_plot" + ".svg", bbox_inches="tight")
plt.close()
