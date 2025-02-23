# Affinity maximization in polarized population

We consider affinity maximization of a polarized population equipped with nonlinear dynamics.

## Main contents
`RecPoliticsPolarized`  
`├── Config/`  
`│   ├── params.yaml`: parameters of the population and the algorithm <br>
`├── Data/`  
`│   ├──`...: store data <br>
`├── Figures/`  
`│   ├──`...: visualization <br>
`├── Models/`  
`│   ├── cl_response.py`: simulate the closed-loop response <br>
`│   ├── population.py`: dynamics and characteristics of the population <br>
`├── Plot/`  
`│   ├── plot_pop_convergence.py`: plot convergence results <br>
`├── Solvers/`  
`│   ├── composite.py`: the proposed algorithm aware of the composite structure <br>
`│   ├── vanilla.py`: the vanilla online stochastic algorithm <br>
`│   ├── tool_funcs.py`: tool functions, e.g., projection <br>
`├── closed_loop_simulation.py`: implement responses & compare algorithms <br>
`├── params_config.py`: generate different combinations of parameters <br>
`├── requirements.txt`: necessary packages

## How to run

- Call closed_loop_simulation.py. The results will be stored in the folders of Data and Figures.  
  If no additional parameter is provided, then params.yaml is loaded.
  Otherwise, params_{i}.yaml is loaded, where i = 1, ...
- Plot/plot_pop_convergence.py allows adjusting the layout of figures based on the stored data in the folder of Data.