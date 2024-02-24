# RL-Gridworld-and-Cartpole

## Gridworld

### Parameterized Policy

Started with \sigma = 0.1 which didn't improve the policy much over time. Increased \sigma to 0.9 which helped sample from a wider distribution, and this resulted in finding better theta that results in near optimal return.

![Performance using various sigma](/GridWorld/parameterize_policy_hill_search.png)


![Performance using optimal parameterized policy](/GridWorld/parameterized_policy_optimal.png)

### Value Iteration

![Performance using greedy policy on value function obtained using value iteration](/GridWorld/value_iteration.png)


![Comparison of optimal policies b/w Parameterized Policy agent and Value Iteration agent](/GridWorld/avg_return_both_agents.png)
## Cartpole

