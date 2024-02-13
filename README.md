# Causal discovery under off-target interventions

This is the accompanying repository of the Accompanying repository for "Causal discovery under off-target interventions". It is available at (coming soon).

To run, execute `./run.sh`. Plots will be saved under `data/figures`. We have included a copy of the produced `figures` sub-directory in here so you may look at the output without running the experiments yourself.

## Executive summary

An instance is defined by an underlying ground truth DAG $G^*$ and $k$ actions $A_1, \ldots, A_k$ with corresponding interventional distributions $D_1, \ldots, D_k$.
We tested on both synthetic and real-world graphs and 3 different classes of interventional distributions.

We compared against 4 baselines: `Random`, `One-shot`, `Coloring`, `Separator`.

`One-shot` tries to emulate non-adaptive interventions while the last two are state-of-the-art on-target search algorithms adapted to the off-target setting.
As `Coloring` and `Separator` were designed specifically for unweighted settings, we test using uniform cost actions despite our off-target search algorithm being able to work with non-uniform action costs.
We also plotted the optimal value of VLP for comparison.

Qualitatively, `Random` and `One-shot` perform visibly worse than the others.
While the adapted on-target algorithms may empirically outperform `Off-Target` sometimes, we remark that our algorithm has provable guarantees even for non-uniform action costs and it is designed to handle worst-case off-target instances.
Since we do not expect real-world causal graphs to be adversarial, it is unsurprising to see that our algorithm performs similarly to `Coloring` and `Separator`.

#### Remark

To properly evaluate adaptive algorithms, one would need data corresponding to all the interventions that these algorithms intend to perform.
Therefore, in addition to observational data, any experimental dataset to evaluate these algorithms should contain interventional data for all possible interventions.
Unfortunately, such real world datasets do not currently exist and thus the state-of-the-art adaptive search algorithms still use synthetic experiments to evaluate their performances.
To slightly mitigate a possible concern of synthetic graphs, we use real-world DAGs from `bnlearn` [Scu10] as our ground truth DAGs $G^*$.

## Graph instances

We tested on synthetic `GNP_TREE` graphs [CS23] of various sizes, and on some real-world graphs from `bnlearn` [Scu10].
We associate a unit-cost action $A_v$ to each vertex $v \in V$ of the input graph.

### Synthetic graphs

For given $n$ and $p$ parameters, the moral `GNP_TREE` graphs are generated in the following way (Description from Appendix F.1.1 of [CS23]):

- Generate a random Erdős-Rényi graph $G(n,p)$.
- Generate a random tree on $n$ nodes.
- Combine their edgesets and orient the edges in an acyclic fashion: orient $u \to v$ whenever vertex $u$ has a smaller vertex numbering than $v$.
- Add arcs to remove v-structures: for every v-structure $u \to v \gets w$ in the graph, we add the arc $u \to w$ whenever vertex $u$ has a smaller vertex numbering from $w$.

We generated `GNP_TREE` graphs with $n \in \\{10, 20, 30, 40, 50\\}$ and $p = 0.1$.
For each $(n,p)$ setting, we generated 10 such graphs.

### Real-world graphs

The `bnlearn` [Scu10] graphs are available at https://www.bnlearn.com/bnrepository/.
In particular, we used the graphical structure of the "Discrete Bayesian Networks" for all sizes: "Small Networks ($< 20$ nodes)", "Medium Networks ($20-50$ nodes)", "Large Networks ($50-100$ nodes)", and "Very Large Networks ($100-1000$ nodes)", and "Massive Networks ($>1000$ nodes)".
Some graphs such as `pigs`, `cancer`, `survey`, `earthquake`, and `mildew` already have fully oriented essential graphs and are thus excluded from the plots as they do not require any interventions.

## Interventional distributions

In our experiments, we associated each vertex $v$ with unit cost and an action $A_v$ with four different possible types of interventional distributions (see below).
The first two are atomic in nature (all actions return a single intervened vertex) while the third is slightly more complicated interventional distribution where multiple vertices may be intervened upon.
Atomic interventional distributions enables a simple way to compute the probability that edge $\\{u,v\\}$ is cut by action $A_i$: it is simply $p^i_u + p^i_v$, where $p^i_v$ is the probability that $v$ is intervened upon when we perform action $A_i$.

The 3 classes of off-target interventions we explored are as follows:

1. $r$-hop  
When taking action $A_v$, $D_v$ samples a uniform random vertex from the closed $r$-hop neighborhood of $v$, including $v$.

2. Decaying with parameter $\alpha$  
When taking action $A_v$, $D_v$ samples a random vertex from a weighted probability distribution $w$ obtained by normalizing the following weight vector: assign weight $\alpha^r$ for all vertices exactly $r$-hops from $v$, where $v$ itself has weight 1.
So, vertices closer to $v$ have higher chance of being intervened upon when we attempt to intervene on $v$.

3. Fat hand with parameter $p$  
When taking action $A_v$, $D_v$ will always intervene on $v$, but will additionally intervene on $v$'s neighbors, each with independent probability $p$. Note that the probability of cutting an edge $\\{u,v\\}$ now is no longer a simple sum of two independent probabilities, but it is still relatively easy to compute in closed-form.

In our experiments, we tested the following 6 settings:
1. $r$-hop with $r = 1$
2. $r$-hop with $r = 2$
3. Decaying with $\alpha = 0.5$
4. Decaying with $\alpha = 0.9$
5. Fat hand with $p = 0.5$
6. Fat hand with $p = 0.9$

## Algorithms

Since our off-target intervention setting has not been studied before from an algorithmic perspective, there is no suitable prior algorithms to compare against.
As such, we propose the following baselines:

#### `Random`

Repeatedly sample actions uniformly at random until the entire DAG is oriented.
This is a natural naive baseline to compare against.


#### `One-shot`

Solve our linear program VLP in the paper on all unoriented edges.
Intepret the optimal vector $X$ of VLP as a probability distribution $p$ over the actions and sample actions according to $p$ until all the unoriented edges are oriented.
One-shot aims to simulate non-adaptive algorithms in the context of off-target interventions: while it can optimally solve VLP (c.f. compute graph separating system), One-shot cannot update its knowledge based on arc orientations that are subsequently revealed.

#### `Coloring` and `Separator`

Two state-of-the-art adaptive on-target intervention algorithms in the literature: `Separator` [CSB22] and `Coloring` [SKDV15].
As these algorithms are not designed for off-target intervention, we need to orient all the edges incident to $v$ to simulate an on-target intervention at $v$.
To do so, we run VLP on the unoriented edges incident to $v$ and interpret the optimal vector $X$ of VLP as a probability distribution $p$ over the actions, then sample actions according to $p$ until all the unoriented edges incident to $v$ are oriented.
Note that this modification provides a generic way to convert any usual intervention algorithm to the off-target setting.

## Experimental results

For each combination of graph instance and interventional distribution, we ran $10$ times and plotted the average with standard deviation error bars.
This is because there is inherent randomness involved when we attempt to perform an intervention.
For synthetic graphs, we also aggregated the performance over all graphs with the same number of nodes $n$ in hopes of elucidating trends with respect to the size of the graph.
As the naive baseline Random incurs significantly more cost than the others, we also plotted all experiments without it.

The optimal value of VLP in <span style="color:blue">blue</span> is an $O(\log n)$ approximation of $\nu(G^*)$.
Our off-target search `Off-Target` is in <span style="color:orange">orange</span>.
`Coloring` is in <span style="color:green">green</span>.
`Separator` is in <span style="color:red">red</span>.
`One-shot` is in <span style="color:purple">purple</span>.
`Random` is in <span style="color:brown">brown</span>.

#### Without `Random`

`GNP_TREE` graphs without `random`:
<p float="middle">
<img src="./figures/synthetic_no_random_10_0.1_['r_hop', [1]]_r_hop.png" alt="1-hop" width="45%"/>
<img src="./figures/synthetic_no_random_10_0.1_['r_hop', [2]]_r_hop.png" alt="2-hop" width="45%"/>
<img src="./figures/synthetic_no_random_10_0.1_['decaying', [0.5]]_decaying.png" alt="Decaying, $\alpha = 0.5$" width="45%"/>
<img src="./figures/synthetic_no_random_10_0.1_['decaying', [0.9]]_decaying.png" alt="Decaying, $\alpha = 0.9$" width="45%"/>
<img src="./figures/synthetic_no_random_10_0.1_['fat_hand', [0.5]]_fat_hand.png" alt="Fat hand, $p = 0.5$" width="45%"/>
<img src="./figures/synthetic_no_random_10_0.1_['fat_hand', [0.9]]_fat_hand.png" alt="Fat hand, $p = 0.9$" width="45%"/>
</p>

`bnlearn` graphs without `random`:
<p float="middle">
<img src="./figures/bnlearn_no_random_10_['r_hop', [1]]_r_hop.png" alt="1-hop" width="45%"/>
<img src="./figures/bnlearn_no_random_10_['r_hop', [2]]_r_hop.png" alt="2-hop" width="45%"/>
<img src="./figures/bnlearn_no_random_10_['decaying', [0.5]]_decaying.png" alt="Decaying, $\alpha = 0.5$" width="45%"/>
<img src="./figures/bnlearn_no_random_10_['decaying', [0.9]]_decaying.png" alt="Decaying, $\alpha = 0.9$" width="45%"/>
<img src="./figures/bnlearn_no_random_10_['fat_hand', [0.5]]_fat_hand.png" alt="Fat hand, $p = 0.5$" width="45%"/>
<img src="./figures/bnlearn_no_random_10_['fat_hand', [0.9]]_fat_hand.png" alt="Fat hand, $p = 0.9$" width="45%"/>
</p>

#### With `Random`

`GNP_TREE` graphs with `random`:
<p float="middle">
<img src="./figures/synthetic_10_0.1_['r_hop', [1]]_r_hop.png" alt="1-hop" width="45%"/>
<img src="./figures/synthetic_10_0.1_['r_hop', [2]]_r_hop.png" alt="2-hop" width="45%"/>
<img src="./figures/synthetic_10_0.1_['decaying', [0.5]]_decaying.png" alt="Decaying, $\alpha = 0.5$" width="45%"/>
<img src="./figures/synthetic_10_0.1_['decaying', [0.9]]_decaying.png" alt="Decaying, $\alpha = 0.9$" width="45%"/>
<img src="./figures/synthetic_10_0.1_['fat_hand', [0.5]]_fat_hand.png" alt="Fat hand, $p = 0.5$" width="45%"/>
<img src="./figures/synthetic_10_0.1_['fat_hand', [0.9]]_fat_hand.png" alt="Fat hand, $p = 0.9$" width="45%"/>
</p>

`bnlearn` graphs with `random`:
<p float="middle">
<img src="./figures/bnlearn_10_['r_hop', [1]]_r_hop.png" alt="1-hop" width="45%"/>
<img src="./figures/bnlearn_10_['r_hop', [2]]_r_hop.png" alt="2-hop" width="45%"/>
<img src="./figures/bnlearn_10_['decaying', [0.5]]_decaying.png" alt="Decaying, $\alpha = 0.5$" width="45%"/>
<img src="./figures/bnlearn_10_['decaying', [0.9]]_decaying.png" alt="Decaying, $\alpha = 0.9$" width="45%"/>
<img src="./figures/bnlearn_10_['fat_hand', [0.5]]_fat_hand.png" alt="Fat hand, $p = 0.5$" width="45%"/>
<img src="./figures/bnlearn_10_['fat_hand', [0.9]]_fat_hand.png" alt="Fat hand, $p = 0.9$" width="45%"/>
</p>

## References

[This paper] Davin Choo, Kirankumar Shiragur, Caroline Uhler. Causal discovery under off-target interventions. International Conference on Artificial Intelligence and Statistics, 2024. Available at (Coming soon).

[Scu10] Marco Scutari. Learning Bayesian networks with the bnlearn R package. Journal of Statistical Software, 2010. Available at https://arxiv.org/pdf/0908.3817.pdf

[SKDV15] Karthikeyan Shanmugam, Murat Kocaoglu, Alexandros G. Dimakis, and Sriram Vishwanath. Learning Causal Graphs with Small Interventions. Advances in Neural Information Processing Systems, 2015. Available at https://arxiv.org/abs/1511.00041

[CSB22] Davin Choo, Kirankumar Shiragur, and Arnab Bhattacharyya. Verification and search algorithms for causal DAGs. Advances in Neural Information Processing Systems, 2022. Available at https://arxiv.org/pdf/2206.15374.pdf

[CS23] Davin Choo and Kirankumar Shiragur. Subset verification and search algorithms for causal DAGs. International Conference on Artificial Intelligence and Statistics, 2023. Available at https://arxiv.org/pdf/2301.03180.pdf.