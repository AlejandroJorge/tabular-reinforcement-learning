import numpy as np
import matplotlib.pyplot as plt

def plot_bandits(bandit_means):
    k = len(bandit_means)
    synthetic_data = [np.random.normal(loc=m, scale=s, size=10_000) for m,s in zip(bandit_means, np.full(k, 1.0))]

    plt.violinplot(synthetic_data, showmeans=True, showmedians=True, showextrema=True)
    plt.xlabel("Bandit")
    plt.ylabel("Reward distribution")
    plt.title("Bandits distribution")

    plt.xticks(range(1,k+1))

    plt.show()

def plot_bandits_true_vs_estimated_reward(Q, bandit_means):
    k = len(bandit_means)
    bar_width = 0.35
    
    bandits = np.arange(k)
    plt.bar(bandits - bar_width/2, bandit_means, bar_width, label="True reward means")
    plt.bar(bandits + bar_width/2, Q, bar_width, label="Estimated rewards")

    plt.legend()
    plt.xlabel("Bandit")
    plt.ylabel("Reward")
    plt.xticks(range(k))
    plt.title("True mean vs estimated rewards for each bandit")

    plt.show()

def plot_bandit_rewards(rewards):
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    plt.title("Agent rewards obtained vs steps")

    plt.show()
    
def plot_bandits_optimal_actions(optimal_actions):
    plt.plot(range(len(optimal_actions)), optimal_actions)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Actions")

    plt.title("Agent % optimal actions vs steps")

    plt.show()

def plot_bandit_rewards_comparison(
    items
):
    x = range(len(items[0][0]))
    for data, label in items:
        plt.plot(x, data, label=label)
    
    plt.xlabel("Steps")
    plt.ylabel("Average reward")

    plt.title("Reward comparison of bandit algorithms")
    plt.legend()

    plt.show()

def plot_bandit_opt_act_comparison(
    items
):
    x = range(len(items[0][0]))
    for data, label in items:
        plt.plot(x, data, label=label)
    
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")

    plt.title("% Optimal action comparison of bandit algorithms")
    plt.legend()

    plt.show()