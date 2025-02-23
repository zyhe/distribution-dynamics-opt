"""
Create a set of hyperparameters for testing on the cluster
"""
import yaml
import itertools

# Define hyperparameter ranges
angle_bd_values = [90]
lambda_p_values = [0.3, 0.4]  # 0.4
sigma_values = [0.4, 0.5]
sz_values = [5e-3]  # 1e-3, 5e-3, 1e-2

# Generate all combinations
combinations = list(itertools.product(angle_bd_values, lambda_p_values, sigma_values, sz_values))

# Fixed parameters
fixed_params = {
    'population': {
        'dim': 20,
        'radi': 1,
        'size_pop': 1000,
        'distri_md': 'uniform'
    },
    'algorithm': {
        'num_sample': 50,
        # 'num_itr': 6000,
        'num_trial': 20
    }
}

# # Mapping from the step size to the number of iterations
# sz_2_num_itr_dict = {
#     1e-3: int(1.5e4),
#     5e-3: 8000,
#     1e-2: 6000
# }

# Mapping from the angle bound to the number of iterations
angle_bd_2_num_itr_dict = {
    70: int(2e4),
    90: 8000,
    100: 8000,
}

# Create YAML files
for i, (angle_bd, lambda_p, sigma, sz) in enumerate(combinations):
    # # Determine num_itr based on sz
    # num_itr = sz_2_num_itr_dict

    # Determine num_itr based on angle_bd
    num_itr = angle_bd_2_num_itr_dict[angle_bd]

    params = {
        'population': {
            'dim': fixed_params['population']['dim'],
            'radi': fixed_params['population']['radi'],
            'size_pop': fixed_params['population']['size_pop'],
            'angle_bd': angle_bd,
            'lambda_p': lambda_p,
            'sigma': sigma,
            'distri_md': fixed_params['population']['distri_md']
        },
        'algorithm': {
            'sz': sz,
            'num_sample': fixed_params['algorithm']['num_sample'],
            'num_itr': num_itr,
            'num_trial': fixed_params['algorithm']['num_trial']
        }
    }
    with open(f'Config/params_{i}.yaml', 'w') as file:
        yaml.dump(params, file)

print(f"Generated {len(combinations)} YAML files.")