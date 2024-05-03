import configparser

# Create a ConfigParser instance
config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

TASK_PROMPT=config['PROMPT']['SYSTEM']
