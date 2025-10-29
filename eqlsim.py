import types
import random
import numpy as np
import matplotlib.pyplot as plt

n_sims = 300
n_raffles = 1000

# takes mean and stdev, returns a value from that normal distribution
def sample_normal(mean, stdev):
    return np.random.normal(loc=mean, scale=stdev)

# here the class strategy is given "coupons" which increment by 1 in value if used, 
# and return to 1 if not used. The "strategies" section will be broken into another file
# and expanded upon if/when i return to this project.
class Strategy():
    def __init__(self, choose_coupon_func, name):
        self.coupons = {1:10e9}
        self.avs = np.zeros((n_sims, n_raffles + 1))
        self.choose_coupon = types.MethodType(choose_coupon_func, self)
        self.name = name

    def adj_coupons(self, value=int, used=bool):
        print(self.coupons)
        self.coupons[value] -= 1

        if self.coupons[value] <= 0:
            del self.coupons[value]

        if used:
            self.coupons[1] += 1
        else:
            if value + 1 in self.coupons:
                self.coupons[value + 1] += 1
            else:
                self.coupons[value + 1] = 1
        
# some trivial strategies. I would like to optimize and find a better strategy, 
# but with limited data of the real sample distribution I have not found it ideal to simulate yet.
def choose_highest(self, profit, entries, winners):
    return max(self.coupons)

# trivially bad strategy
def choose_lowest(self, profit, entries, winners):
    return min(self.coupons)

# this strategy has a tendency to both reduce variance and skew variance
# upwards without compromising EV compared to choose highest
def find_balanced_coupon(self, profit, entries, winners):
    def ev_of(coupon):
        k = coupon
        p_win = 1 - ((entries - winners)/entries)**k
        return p_win * profit
    
    k = max(self.coupons)
    ev = ev_of(k)
    adjev = ev * 0.975
    # choosing a lower adjusted EV quickly results in lower profit, 
    # but slightly reducing ev has shown to be largely beneficial

    # checks to see if another coupon will provide at least 97.5%
    # of the EV that out "best" coupon currently provides
    for coupon in sorted(self.coupons.keys(), reverse=True):
        if ev_of(coupon) > adjev:
            k = coupon
        else:
            return k
    return k

strat1 = Strategy(choose_highest, "Highest")
strat2 = Strategy(find_balanced_coupon, "Balanced")
strategies = [strat1, strat2]

# some very rough estimates i have for the raffle data in usd.
mean_profit = 1000
stdev_profit = 1500
mean_entries = 550000
stdev_entries = 10000
mean_winners = 5000
stdev_winners = 2000

eps = 1e-6

# runs a simulation 50 times, and iterates over every strategy per simulation
for x in range (n_sims):
    i = 1

    # runs n raffles, skipping and not incrementing if profit is <= 0.
    while i < n_raffles + 1:
        profit = sample_normal(mean_profit,stdev_profit)
        if profit <= 0:
            continue

        z = (profit - mean_profit) / stdev_profit
        z_entries = z * stdev_entries + mean_entries
        z_winners = max(500, mean_winners - z * stdev_winners)

        # randomizes num entries and num winners
        safe_div = np.where(np.abs(z) > eps, np.abs(z), 1.0)
        r_entries = sample_normal(z_entries, stdev_entries / safe_div)
        r_winners = sample_normal(z_winners, stdev_winners / safe_div)

        # runs all the strategies for the sample raffle
        for strategy in strategies:
            k = strategy.choose_coupon(profit, z_entries, z_winners)

            p_win = 1 - ((r_entries - r_winners)/r_entries)**k
            p_win = max(0, min(1, p_win))

            # determines if the raffle is won or not
            is_win = (random.random() < p_win)
            av =  is_win * profit

            strategy.adj_coupons(k, is_win)

            # adds the profit to every point
            if i == 0:
                strategy.avs[x][i] = av
            else:
                strategy.avs[x][i] = av + strategy.avs[x][i-1]

        i += 1


plt.figure(figsize=(12,6))

colors = ['#6ba4ff',"#88fba1","#ffaf7d"]

for c, strat in enumerate(strategies):
    # plot every sim lightly
    for sim_idx in range(n_sims):
        plt.plot(strat.avs[sim_idx], alpha=0.15, color=colors[c])
    # plot median or mean trajectory prominently with a label
    median_traj = np.median(strat.avs, axis=0)
    plt.plot(median_traj, color=colors[c], label=strat.name, linewidth=2)

plt.xlabel("Simulation index")
plt.ylabel("Cumulative value")
plt.title("Cumulative EV vs AV per Strategy")
plt.legend()
plt.show()

# boxplot of total AVs for comparison
plt.figure(figsize=(8,5))
final_totals = [strat.avs[:, -1] for strat in strategies]  # list of 1D arrays
plt.boxplot(final_totals, labels=[s.name for s in strategies])
plt.ylabel("Total Actual Value (AV) at end of run")
plt.title("Distribution of AV by Strategy")
plt.show()