# config.yaml for the streamlit app is just a truncated version of what the 
# Open Creativity Scoring API uses.
import yaml
import os

# Keys to keep in each model entry
KEYS_TO_KEEP = [
    'name', 
    'description', 
    'short-description', 
    'format', 
    'style',
    'languages', 
    'tasks', 
    'debug', 
    'deprecated', 
    'recommended', 
    'production'
]

def truncate_config():
    # Read the original config file
    config_path = '../ocs-online/config.yaml'
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract only the llmmodels section
    llmmodels = config.get('llmmodels', [])
    
    # Create a new config with only the llmmodels (truncated)
    truncated_config = {'llmmodels': []}
    
    for model in llmmodels:
        # Skip models that are deprecated or not in production
        if model.get('deprecated', False) or not model.get('production', True):
            continue
            
        # Create a new model entry with only the specified keys
        truncated_model = {}
        for key in KEYS_TO_KEEP:
            if key in model:
                truncated_model[key] = model[key]
        
        # Add the truncated model to the new config
        truncated_config['llmmodels'].append(truncated_model)
    
    
    # Save the truncated config to the streamlit folder
    output_path = os.path.join('.', 'config.yaml')
    with open(output_path, 'w') as file:
        yaml.dump(truncated_config, file, default_flow_style=False, sort_keys=False)
    
    print(f"Truncated config saved to {output_path}")

if __name__ == "__main__":
    truncate_config()