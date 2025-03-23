import os
import re
import sqlite3
import json
import numpy as np
import logging
from datetime import datetime
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate

# -----------------------------------------
# Logging configuration
# -----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------
# Load environment variables
# -----------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set!")

# -----------------------------------------
# Define the update_patient_diagnosis tool for function calling.
# -----------------------------------------
tools = [{
    "name": "update_patient_diagnosis",
    "description": "Update patient's diagnostic criteria based on reported symptoms. Requires patient_id and symptoms.",
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {"type": "integer", "description": "Unique patient identifier"},
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of observed symptoms in the patient"
            }
        },
        "required": ["patient_id", "symptoms"]
    }
}]

# -----------------------------------------
# Initialize vector database (Chroma) for conversation retrieval.
# -----------------------------------------
persist_directory = "./chroma_db"
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# -----------------------------------------
# Function to fetch patient name from diagnosis.db given patient_id.
# -----------------------------------------
def get_patient_name(patient_id: int) -> str:
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM patients WHERE id = ?", (patient_id,))
            result = cursor.fetchone()
            return result[0] if result else "Unknown"
    except Exception as e:
        return "Unknown"

# -----------------------------------------
# Initialize (or create) the conversation history database.
# -----------------------------------------
def init_conversation_db():
    conn = sqlite3.connect('conversation.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            timestamp TEXT,
            sender TEXT,
            message TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_message(patient_id: int, sender: str, message: str):
    conn = sqlite3.connect('conversation.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO conversation_history (patient_id, timestamp, sender, message) VALUES (?, ?, ?, ?)",
        (patient_id, timestamp, sender, message)
    )
    conn.commit()
    conn.close()

def load_conversation_history(patient_id: int) -> str:
    """Load all messages for a given patient_id and return them as a single string."""
    conn = sqlite3.connect('conversation.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sender, message FROM conversation_history WHERE patient_id = ? ORDER BY id ASC",
        (patient_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    history = "\n".join(f"{sender}: {message}" for sender, message in rows)
    return history

# Helper function to parse conversation history string into list of (human, ai) pairs.
def parse_conversation_history(conv_str: str):
    """
    Converts a conversation history string in the format:
      "Psychiatrist: Hi\nAssistant: Hello, how can I help?\nPsychiatrist: What is NPD?"
    into a list of tuples like:
      [("human", "Hi"), ("ai", "Hello, how can I help?"), ("human", "What is NPD?")]
    """
    lines = conv_str.strip().split("\n")
    history = []
    for line in lines:
        if line.startswith("Psychiatrist:"):
            content = line.replace("Psychiatrist:", "").trim()
            history.append(("human", content))
        elif line.startswith("Assistant:"):
            content = line.replace("Assistant:", "").trim()
            history.append(("ai", content))
        # You can add additional conditions for other roles if needed.
    return history

# Initialize conversation database on startup.
init_conversation_db()

# -----------------------------------------
# Set up LangChain for the conversation.
# -----------------------------------------
system_msg = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant called MindSeek that remembers details from the conversation. "
    "The user is a psychiatrist, not the patient. Always refer to the patient using the provided patient details. "
    "You must not reveal that you're an OpenAI model or GPT; say instead that you're an assistant with proprietary technology. "
    "Your hobby is helping, and you strive to provide accurate and helpful information. "
    "You are augmented with external knowledge sources, especially from the DSM-5. "
    "If DSM-5 does not provide an answer, use general mental health knowledge to help the user. "
    "When the user provides any symptoms, you MUST automatically call the update_patient_diagnosis function to update the patient's diagnostic criteria. "
    "Also, if the user asks about the patient's name, provide it. "
    "Remember: The user is a psychiatrist seeking assistance with their patient's case."
    "Please ALWAYS output your answer in Markdown format. MAKE SURE it's PROPERLY FORMATTED."
)

human_msg = HumanMessagePromptTemplate.from_template(
    "Psychiatrist: {psychiatrist_name}\n"
    "Patient ID: {patient_id}\n"
    "Patient Name: {patient_name}\n"
    "Current Time: {current_time}\n"
    "Conversation History:\n{chat_history}\n"
    "Context:\n{context}\n"
    "User (as Psychiatrist): {question}\n"
    "Assistant:"
)
qa_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0.7,
    functions=tools,
    function_call="auto"
)

condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the following conversation history and a follow-up question, "
        "rephrase the follow-up question to be a standalone question.\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "Follow-up question: {question}\n\n"
        "Standalone question:"
    )
)
question_generator_chain = LLMChain(llm=llm, prompt=condense_prompt)

# Remove in-memory conversation memory to avoid cross-user leakage.
# Instead, we always load the full conversation history from SQLite.
qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

conversational_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=qa_chain,
    memory=None,  # No in-memory memory.
    return_source_documents=True
)

relevance_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "Decide if the following question is related to mental health, psychology, "
        "or DSM-5 disorders. If the query is about weather, call the weather function automatically. "
        "Answer with 'yes' or 'no' only.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)
relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)

# -----------------------------------------
# Diagnosis Database Initialization (for disorders, patients, criteria).
# -----------------------------------------
def init_diagnosis_db():
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS disorders
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 criteria TEXT NOT NULL)
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patient_criteria
                (patient_id INTEGER,
                 disorder_id INTEGER,
                 criterion TEXT,
                 met BOOLEAN,
                 PRIMARY KEY (patient_id, disorder_id, criterion))
            ''')

            cursor.execute("SELECT COUNT(*) FROM disorders")
            count = cursor.fetchone()[0]
            if count == 0:
                sample_disorders = [
                    ("Major Depressive Disorder", [
                        "Depressed mood", 
                        "Markedly diminished interest", 
                        "Significant weight change", 
                        "Insomnia or hypersomnia", 
                        "Fatigue or loss of energy", 
                        "Feelings of worthlessness"
                    ]),
                    ("Bipolar I Disorder", [
                        "At least one manic episode", 
                        "Inflated self-esteem or grandiosity", 
                        "Decreased need for sleep", 
                        "More talkative than usual", 
                        "Flight of ideas", 
                        "Increase in goal-directed activity"
                    ])
                ]
                for name, criteria in sample_disorders:
                    cursor.execute("INSERT INTO disorders (name, criteria) VALUES (?, ?)",
                                   (name, json.dumps(criteria)))
            conn.commit()
            logger.info("Diagnosis database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing diagnosis database: {e}")

init_diagnosis_db()

# -----------------------------------------
# Pydantic models for Diagnosis API.
# -----------------------------------------
class SymptomUpdate(BaseModel):
    patient_id: int
    symptoms: list[str]

class PatientCreate(BaseModel):
    name: str

# -----------------------------------------
# Pydantic models for Conversation API.
# -----------------------------------------
class MessageRequest(BaseModel):
    message: str
    patient_id: int
    psychiatrist_name: str = "Unknown"

class MessageResponse(BaseModel):
    answer: str
    retrieval_info: str = ""

# -----------------------------------------
# Create FastAPI app and endpoints.
# -----------------------------------------
app = FastAPI(title="Integrated Diagnosis and Conversation API",
              description="API for managing diagnosis and conversation for mental health.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend's domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Diagnosis Endpoints -----
@app.post("/create_patient")
async def create_patient(patient: PatientCreate):
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO patients (name) VALUES (?)", (patient.name,))
            conn.commit()
            patient_id = cursor.lastrowid
            logger.info(f"Created patient {patient_id} with name {patient.name}")
            return {"patient_id": patient_id}
    except Exception as e:
        logger.error(f"Error creating patient: {e}")
        raise HTTPException(status_code=500, detail="Error creating patient")

@app.post("/update_criteria")
async def update_criteria(update: SymptomUpdate):
    logger.info(f"Updating criteria for patient {update.patient_id}")
    logger.info(f"Symptoms: {update.symptoms}")
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, criteria FROM disorders")
            disorders = cursor.fetchall()
            
            # Precompute embeddings for each criterion per disorder
            disorder_criteria_embeddings = {}
            for disorder_id, criteria_json in disorders:
                try:
                    criteria_list = json.loads(criteria_json)
                except Exception as e:
                    logger.error(f"Error parsing JSON for disorder {disorder_id}: {e}")
                    continue  # Skip if JSON parsing fails
                criterion_embeds = []
                for criterion in criteria_list:
                    crit_emb = np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(criterion))
                    criterion_embeds.append((criterion, crit_emb))
                disorder_criteria_embeddings[disorder_id] = criterion_embeds
                logger.info(f"Precomputed embeddings for disorder {disorder_id}")
            
            # Process each symptom provided by the user.
            for symptom in update.symptoms:
                symptom_emb = np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(symptom))
                for disorder_id, criterion_list in disorder_criteria_embeddings.items():
                    for criterion, crit_emb in criterion_list:
                        norm_symptom = np.linalg.norm(symptom_emb)
                        norm_crit = np.linalg.norm(crit_emb)
                        if norm_symptom == 0 or norm_crit == 0:
                            continue
                        sim = np.dot(symptom_emb, crit_emb) / (norm_symptom * norm_crit)
                        logger.info(f"Similarity between '{symptom}' and '{criterion}': {sim}")
                        if sim >= 0.5:  # SIMILARITY_THRESHOLD is 0.5
                            cursor.execute('''
                                INSERT OR REPLACE INTO patient_criteria
                                (patient_id, disorder_id, criterion, met)
                                VALUES (?, ?, ?, ?)
                            ''', (update.patient_id, disorder_id, criterion, True))
            conn.commit()
            return {"status": "updated"}
    except Exception as e:
        logger.error(f"Error updating criteria: {e}")
        raise HTTPException(status_code=500, detail="Error updating criteria")

@app.get("/diagnosis_summary")
async def diagnosis_summary(patient_id: int):
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.name, pc.criterion
                FROM patient_criteria pc
                JOIN disorders d ON d.id = pc.disorder_id
                WHERE pc.patient_id = ? AND pc.met = 1
            ''', (patient_id,))
            rows = cursor.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="No diagnosis data found for this patient")
            summary = {}
            for disorder_name, criterion in rows:
                summary.setdefault(disorder_name, []).append(criterion)
            logger.info(f"Diagnosis summary for patient {patient_id}: {summary}")
            return {"patient_id": patient_id, "diagnosis_summary": summary}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving diagnosis summary: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving diagnosis summary")

# ----- Conversation Endpoint -----
@app.post("/api/message", response_model=MessageResponse)
async def handle_message(req: MessageRequest):
    print(req)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_id = req.patient_id
    patient_name = get_patient_name(patient_id) if patient_id else "Unknown"
    psychiatrist_name = req.psychiatrist_name or "Unknown"
    user_query = req.message

    # Store the user's message in the conversation history.
    store_message(patient_id, "Psychiatrist", user_query)

    # Load the full conversation history from SQLite.
    const_history = load_conversation_history(patient_id) if patient_id else ""
    # Parse the conversation history into structured format:
    parsed_history = parse_conversation_history(const_history)

    # Process the message with the LLM.
    raw_response = llm.invoke(user_query)
    function_call = raw_response.additional_kwargs.get("function_call")
    
    if function_call and function_call.get("name") == "update_patient_diagnosis":
        try:
            args = json.loads(function_call.get("arguments", "{}"))
            patient_id = args.get("patient_id") or patient_id
            if patient_id is None:
                raise ValueError("Patient ID is not provided.")
            response = requests.post(
                "http://localhost:8000/update_criteria",
                json={"patient_id": patient_id, "symptoms": args["symptoms"]}
            )
            with sqlite3.connect('diagnosis.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''SELECT d.name, COUNT(pc.criterion) as met_count 
                                  FROM patient_criteria pc
                                  JOIN disorders d ON pc.disorder_id = d.id
                                  WHERE pc.patient_id = ?
                                  GROUP BY d.id''', (patient_id,))
                diagnoses = cursor.fetchall()
            diagnosis_text = "\n".join([f"{name}: {count} criteria met" for name, count in diagnoses])
            answer = f"Diagnostic criteria updated. \nCurrent matches:\n{diagnosis_text}"
            retrieval_info = ""
        except Exception as e:
            answer = f"Error updating diagnosis: {str(e)}"
            retrieval_info = ""
    else:
        relevance_result = relevance_chain.invoke({"question": user_query})
        is_relevant = relevance_result["text"].strip().lower()
        if is_relevant == "yes":
            context = f"Patient ID: {patient_id}, Patient Name: {patient_name}" if patient_id else ""
            result = conversational_chain({
                "chat_history": parsed_history,
                "question": user_query,
                "current_time": current_time,
                "context": context,
                "patient_id": patient_id or "unknown",
                "patient_name": patient_name,
                "psychiatrist_name": psychiatrist_name,
            })
            answer = result["answer"]
            if result.get("source_documents") and len(result["source_documents"]) > 0:
                pages_by_source = {}
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page")
                    if page is not None:
                        pages_by_source.setdefault(source, set()).add(page)
                    else:
                        pages_by_source.setdefault(source, set()).add("unknown")
                parts = []
                for src, pages in pages_by_source.items():
                    try:
                        sorted_pages = sorted(pages, key=lambda x: int(x))
                    except Exception:
                        sorted_pages = sorted(pages)
                    pages_str = ", ".join(str(p) for p in sorted_pages)
                    parts.append(f"{src} (pages: {pages_str})")
                retrieval_info = "Retrieved from " + ", ".join(parts)
            else:
                retrieval_info = "No relevant information found in DSM-5. Answer generated from general knowledge."
        else:
            answer = qa_chain.run({
                "chat_history": parsed_history,
                "patient_id": patient_id,      
                "patient_name": patient_name,  
                "psychiatrist_name": psychiatrist_name,
                "context": "",
                "question": user_query,
                "current_time": current_time,
                "input_documents": [],
            })
            retrieval_info = "None"

    # Store the assistant's response in the conversation history.
    store_message(patient_id, "Assistant", answer)

    print(answer)

    return MessageResponse(answer=answer, retrieval_info=retrieval_info)

# Helper: Parse conversation history string to structured list.
def parse_conversation_history(conv_str: str):
    """
    Converts a conversation history string like:
      "Psychiatrist: Hi\nAssistant: Hello, how can I help?\nPsychiatrist: What is NPD?"
    into a list of tuples: [("human", "Hi"), ("ai", "Hello, how can I help?"), ("human", "What is NPD?")]
    """
    lines = conv_str.strip().split("\n")
    parsed = []
    for line in lines:
        if line.startswith("Psychiatrist:"):
            content = line.replace("Psychiatrist:", "").strip()
            parsed.append(("human", content))
        elif line.startswith("Assistant:"):
            content = line.replace("Assistant:", "").strip()
            parsed.append(("ai", content))
    return parsed

# -----------------------------------------
# Run the app with Uvicorn if executed directly.
# -----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
