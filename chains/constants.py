from langchain.chains import ConversationChain
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
)
from langchain_openai import OpenAI

from knowledge.faiss_db import db
from .prompts import *

llm = OpenAI(temperature=0)


# Here it is by default set to "AI"
conversation_chain = ConversationChain(
    prompt=TEMPLATE_CONFIG["conversation"],
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)



from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_TEMPLATE = """.
<context>
{context}
</context>
Since you're only a router agent, you must only provide your answer in the following format:
```
Suggested Genie to talk to: {{name of the genie}}
URL of the Genie: {{url of the genie}}
{{explanation of why this genie is the best choice by analysis of user needs}}
For the following user query, decide which genie agent is the most appropriate agent for. Then, cites the sources. The query is as follows:
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(
    llm, question_answering_prompt)

### chains

from langchain.chains.question_answering import load_qa_chain

qa_chain = load_qa_chain(llm=llm)

# conversation = ConversationChain(
#     prompt=PROMPT_HISTORY,
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferMemory(
#         ai_prefix="AI Assistant"),
#     context=db.as_retriever()
# )

# advance retriever chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def retriever_chain(retriever):
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    