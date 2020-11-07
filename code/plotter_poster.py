import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

x_points = np.arange(100, 200000, 100)
y_points = np.zeros_like(x_points)
LABELS = ['ucb']
in_file = '../results/i2-out.txt'
bandit_data = pd.read_csv(in_file, sep=", ", header=None)
bandit_data.columns = ["algo", "rs", "eps", "horizon", "REG", "EXP_REG"]
plt.figure(figsize=(5.2, 3*5.2/4.0))

# Plot results for ucb
ucb = bandit_data.loc[bandit_data["algo"] == "ucb"]

# Get data points for each algorithm
for i in range(len(y_points)):
	y_points[i] = ucb.loc[ucb["horizon"] == x_points[i]]["EXP_REG"].mean()
print(y_points)
points = np.arange(len(y_points))
plt.plot(x_points, 3.36*points - y_points, color='r', linewidth=3)
y_line = np.repeat(3.36, len(y_points))
plt.plot(x_points, y_line)

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))
plt.xticks(np.arange(0, max(x_points)+1, 50000))
plt.xlabel("Number of rounds T", fontweight="bold")
plt.ylabel("Cumulative Regret", fontweight="bold")
plt.title("Instance 3: UCB vs. CUCB vs. U-CUCB", fontweight="bold")
plt.yticks()
plt.show()
