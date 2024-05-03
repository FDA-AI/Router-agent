from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_core.prompts.prompt import PromptTemplate
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

from chain.chat import chat
# Define the data model for user input
class UserInput(BaseModel):
    message: str

env_success = load_dotenv()
assert env_success

app = FastAPI(title="Router API")

@app.post("/chat/", response_model=dict)
def chat(user_input: UserInput):
    try:
        return JSONResponse(content={
            "response": chat(input=user_input.message)
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a get route for root to easily verify the API is running
@app.get("/")
def read_root():
    return {"message": "Bio Chat API is running"}

