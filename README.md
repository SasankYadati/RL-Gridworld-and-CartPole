# RL-Gridworld-and-Cartpole

## Gridworld

### Parameterized Policy

Started with $\sigma = 0.1$ which didn't improve the policy much over time. Increased $\sigma$ to 0.9 which helped sample $\theta$ from a wider distribution, and this resulted in finding better $\theta$ that results in near optimal return.

![Performance using various sigma](/GridWorld/parameterize_policy_hill_search.png)


![Performance using optimal parameterized policy](/GridWorld/parameterized_policy_optimal.png)

### Value Iteration

It took about 50 iterations to converge using a $\epsilon$ (error-threshold) of 1e-6.

![Performance using greedy policy on value function obtained using value iteration](/GridWorld/value_iteration.png)


![Comparison of optimal policies b/w Parameterized Policy agent and Value Iteration agent](/GridWorld/avg_return_both_agents.png)

## Cartpole

### Parameterized Gradient Ascent

Tried three different $\sigma$ values ```(0.1, 0.3, 0.9)```. $\sigma=0.9$ gives the best convergence both in terms of reward and speed of convergence.

![Performance using various sigma](/CartPole/Parameterized%20Policy%20Search.png)

### Cross Entropy Method
With $K=10$, $K_{\eps}=3$, $\eps=0.99$ converged to the optimal return of 1000 in about 80 iterations. At each iteration, we keep the best theta average of the top $K_{\eps}$ candidates.