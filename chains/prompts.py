from langchain_core.prompts.prompt import PromptTemplate

import configparser

# Create a ConfigParser instance
config = configparser.ConfigParser()
# Read the config.ini file
config.read('config.ini')
TEMPLATE_CONFIG=config["TEMPLATE"]

LONGEVITY_GENIE = PromptTemplate(
    input_variables=["context", "question"], 
    template=TEMPLATE_CONFIG["longevity_genie"]
)
CONVERSATION = PromptTemplate(
    input_variables=["question", "history"], 
    template=TEMPLATE_CONFIG["conversation"]
)

# default: 3 variables 
DEFAULT = PromptTemplate(
    input_variables=["context", "question", "history"],
    template=TEMPLATE_CONFIG["default"]
)