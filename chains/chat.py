from chains.constants import conversation, retriever_chain, document_chain
from knowledge.faiss_db import db, data, retrieve_similar
# Now we can override it and set it to "AI Assistant"
from typing import Optional, Union, Dict
from dotenv import load_dotenv

from utils import load_yml

CHAT_CONFIG=config["CHAT"]

env_success = load_dotenv()
assert env_success

def simple_chat(user_input):
    return conversation.predict(input=user_input)

def retrieve_chat(user_input: str):
    retriever_chat = retriever_chain(db.as_retriever())
    return retriever_chat.invoke(user_input)

from langchain_core.messages import HumanMessage
    
def doc_chat(user_input:str):
    # data = retrieve_similar(user_input)
    # print(f"context:{data}")
    resp = document_chain.invoke(
        {
            "context": data,
            "messages": [
                HumanMessage(content=user_input)
            ],
        }
    )
    return resp

def chat(user_input: str | Dict):
    match CHAT_CONFIG["method"].lower():
        case "default" | "conversation":
            return simple_chat(user_input)
        case "document-only":
            return doc_chat(user_input)
    return 
    
def main():
    # Start a loop to continually ask for user input and respond
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['/exit', '/q']:
            print("Exiting chat...")
            break
        resp = chat(user_input)
        print("AI Assistant:", resp)
        

# Run the chat function
if __name__ == "__main__":
    main()