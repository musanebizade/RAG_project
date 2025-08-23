import os
import json
import asyncio
from dotenv import load_dotenv
import boto3
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ----------------------
# Load env variables
# ----------------------
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ID = os.getenv("MODEL_ID")

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

# ----------------------
# Clients
# ----------------------
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

kb_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# ----------------------
# Pydantic models
# ----------------------
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# ----------------------
# LLM helper
# ----------------------
def generate_llm_answer(prompt: str, max_tokens: int = 1024, temperature: float = 0.5):
    """Send prompt to the LLM and get a structured response"""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "system": "You are a helpful assistant."
    }

    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )

    result_bytes = response["body"].read()
    result_json = json.loads(result_bytes.decode("utf-8"))
    if result_json.get("content"):
        return result_json["content"][0].get("text", "")
    return "No response from model."

# ----------------------
# Knowledge Base helpers
# ----------------------
def retrieve_from_kb(query: str, top_k: int = 3):
    """Query the knowledge base and return top results"""
    req = {
        "knowledgeBaseId": KNOWLEDGE_BASE_ID,
        "retrievalQuery": {"text": query},
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {"numberOfResults": top_k}
        },
    }
    response = kb_client.retrieve(**req)
    candidates = response.get("retrievalResults", [])
    docs = [
        f"Document {i+1}: {doc['content']['text']}"
        for i, doc in enumerate(candidates)
        if "content" in doc and "text" in doc["content"]
    ]
    return "\n\n".join(docs)

def generate_rag_answer(user_query: str, conversation_history: list):
    """Combine KB retrieval with LLM generation, considering conversation history"""
    kb_context = retrieve_from_kb(user_query)
    
    # Format conversation history - fix the f-string issue
    newline = "\n"  # Extract newline to variable
    conversation_text = newline.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    
    # Build the prompt with formatting instructions
    prompt = f"""
### Knowledge Base:
{kb_context}

### Conversation History:
{conversation_text}

### User Query:
{user_query}

### Response Instructions:
1. Provide clear, well-structured answers using normal text formatting.
2. Use simple paragraphs separated by line breaks.
3. If listing items, use simple bullet points with dashes (-) or numbers.
4. Do NOT use markdown headers (# ## ###) or excessive bold formatting.
5. Keep the text readable with normal font weight.

### Response:
"""
    return generate_llm_answer(prompt)

# ----------------------
# Streaming helper
# ----------------------
async def stream_generator(user_query: str, conversation_history: list):
    answer = generate_rag_answer(user_query, conversation_history)
    for word in answer.split():
        yield word + " "
        await asyncio.sleep(0.05)

# ----------------------
# API endpoints
# ----------------------
@app.post("/rag/query")
def rag_query_endpoint(request: ChatRequest):
    try:
        # Extract conversation history from messages (excluding the last user message)
        conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages[:-1]]
        
        user_message = request.messages[-1].content

        # Call the RAG model to get the response
        answer = generate_rag_answer(user_message, conversation_history)

        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}

@app.post("/rag/stream")
async def rag_stream_endpoint(request: ChatRequest):
    # Extract conversation history from messages (excluding the last user message.)
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages[:-1]]
    
    user_message = request.messages[-1].content

    return StreamingResponse(stream_generator(user_message, conversation_history), media_type="text/plain")