ICML 2020 -- Policy Teaching via Environment Poisoning: Training-time Adversarial Attacks against Reinforcement Learning

## Prerequisites:
```
Python3
Matplotlib
Numpy
Scipy
Cvxpy
Itertools
```

## Running the code
To get results, you will need to run the following scripts:

## For the Chain environment

### For online attack
```
python teaching_online.py
```

### For offline attacks when varying parameter $\overline{R}(s_0, .)$
```
python teaching_offline_vary_c.py
```

### For offline attack when varying parameter $\epsilon$
```
python teaching_offline_vary_eps.py
```

### To see how long it takes to solve P1, P2, P3 and P4 problems when |S|=4, |S|=10, |S|=50 and |S|=100 run:
```
python teaching_time_table.py
```
## ==========================================

## For the Gridworld environment

### For online attack 
```
python teaching_online_grid.py
```

### For offline attacks when varying parameter $\overline{R}(s_0, .)$ 
```
python teaching_offline_vary_c_grid.py
```

### For offline attack when varying parameter $\epsilon$
```
python teaching_offline_vary_eps_grid.py
```

### Results

After running the above scripts, new plots will be created in plots/env_chain or in plots/env_grid directory accordingly.

In the __main__ function, the variable number_of_iterations denotes the number of runs used to average the results. Set a smaller number for faster execution.
