import yaml

# Define the path to your YAML configuration file
def load_yml(yaml_file_path):
  # Open the YAML file and load its content
  with open(yaml_file_path, 'r') as file:
      config = yaml.safe_load(file)
  return config