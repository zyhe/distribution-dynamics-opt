# Performance optimization given discrete choice distributions 

We consider decision-making given discrete distributions evolving in the probability simplex.

`RecHedge`  
`├── Config/`  
`│   ├── params.yaml`: parameters of the population and the algorithm <br>
`├── Data/`  
`│   ├──`...: store data <br>
`├── Figures/`  
`│   ├──`...: visualization <br>
`├── Models/`  
`│   ├── distribution_dynamics.py`: implement the distribution dynamics of the user <br>
`├── Plot/`  
`│   ├── plot_pop_convergence.py`: plot convergence results <br>
`├── Solvers/`  
`│   ├── composite.py`: the proposed algorithm aware of the composite structure <br>
`│   ├── vanilla.py`: the vanilla online stochastic algorithm <br>
`│   ├── vanilla_gf.py`: the vanilla derivative-free algorithm <br>
`│   ├── tool_funcs.py`: tool functions, e.g., projection <br>
`├── closed_loop_response.py`: implement responses & compare algorithms <br>
`├── params_config.py`: generate different combinations of parameters <br>
`├── requirements.txt`: necessary packages

## How to run

- Call closed_loop_response.py. The results will be stored in the folders of Data and Figures.  
  If no additional parameter is provided, then params_3.yaml is loaded.
  Otherwise, params_{i}.yaml is loaded, where i = 0, 1, ...
- Plot/plot_pop_convergence.py allows adjusting the layout of figures based on the stored data in the folder of Data.  
