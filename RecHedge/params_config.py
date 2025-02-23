import yaml
from pathlib import Path
from itertools import product

# Base configuration with fixed parameters
base_config = {
    'dynamics': {
        'lambda1': 0.2,
        'lambda2': 0.5,
        'epsilon': 0.5,
    },
    'algorithm': {
        'sz': 0.5,
        'num_itr': 6e3,
        'sz_gf': 0.05,  # This will be updated
        'delta': 2,     # This will be updated
        'num_trial': 20,
    },
    'problem': {
        'dim': 100,
        'bd_dec': 5,
        'budget': 250,
        'w_entropy': 0.1,
    }
}

# Lists of values for sz_gf and delta
sz_gf_values = [0.05, 0.1] # [1e-4, 1e-5] #[0.05, 0.025, 0.01, 0.005]
delta_values = [1, 2] # [1e-3, 1e-4] #[0.1, 0.5, 1, 2]

# Directory to save the generated YAML files
output_dir = Path('Config')
output_dir.mkdir(exist_ok=True)

# Specify all possible combinations
combinations = list(product(sz_gf_values, delta_values))

# Generate YAML files using itertools.product to create combinations
for i, (sz_gf, delta) in enumerate(combinations):
    config = base_config.copy()
    config['algorithm']['sz_gf'] = sz_gf
    config['algorithm']['delta'] = delta

    filename = f'params_{i}.yaml'
    filepath = output_dir / filename

    with open(filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

print(f'Generated {len(combinations)} YAML files.')