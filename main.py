import os
import re
import sqlite3
import json
import numpy as np
import logging
from datetime import datetime
import requests

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set!")

tools = [
    {
        "type": "function",
        "function": {
            "name": "update_patient_diagnosis",
            "description": "Update patient's diagnostic criteria based on reported symptoms. Requires patient_id and symptoms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        "description": "Unique patient identifier"
                    },
                    "symptoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of observed symptoms in the patient"
                    }
                },
                "required": ["patient_id", "symptoms"]
            }
        }
    }
]

persist_directory = "./chroma_db"
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

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
            content = line.replace("Psychiatrist:", "").strip()
            history.append(("human", content))
        elif line.startswith("Assistant:"):
            content = line.replace("Assistant:", "").strip()
            history.append(("ai", content))
    return history

def update_criteria_logic(patient_id: int, symptoms: List[str]) -> None:
    logger.info(f"Updating criteria for patient {patient_id} with symptoms: {symptoms}")
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, criteria FROM disorders")
            disorders = cursor.fetchall()
            
            # Precompute embeddings for each criterion per disorder.
            disorder_criteria_embeddings = {}
            for disorder_id, criteria_json in disorders:
                try:
                    criteria_list = json.loads(criteria_json)
                except Exception as e:
                    logger.error(f"Error parsing JSON for disorder {disorder_id}: {e}")
                    continue
                criterion_embeds = []
                for criterion in criteria_list:
                    crit_emb = np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(criterion))
                    criterion_embeds.append((criterion, crit_emb))
                disorder_criteria_embeddings[disorder_id] = criterion_embeds
                logger.info(f"Precomputed embeddings for disorder {disorder_id}")
            
            # Process each symptom.
            for symptom in symptoms:
                symptom_emb = np.array(OpenAIEmbeddings(model="text-embedding-3-large").embed_query(symptom))
                for disorder_id, criterion_list in disorder_criteria_embeddings.items():
                    for criterion, crit_emb in criterion_list:
                        norm_symptom = np.linalg.norm(symptom_emb)
                        norm_crit = np.linalg.norm(crit_emb)
                        if norm_symptom == 0 or norm_crit == 0:
                            continue
                        sim = np.dot(symptom_emb, crit_emb) / (norm_symptom * norm_crit)
                        logger.info(f"Similarity between '{symptom}' and '{criterion}': {sim}")
                        if sim >= 0.5:  # similarity threshold
                            cursor.execute('''
                                INSERT OR REPLACE INTO patient_criteria
                                (patient_id, disorder_id, criterion, met)
                                VALUES (?, ?, ?, ?)
                            ''', (patient_id, disorder_id, criterion, True))
            conn.commit()
    except Exception as e:
        logger.error(f"Error updating criteria: {e}")
        raise

init_conversation_db()

system_msg = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant called MindSeek that remembers details from the conversation. "
    "The user is a psychiatrist, not the patient. Always refer to the patient using the provided patient details. "
    "You must not reveal that you're an OpenAI model or GPT; say instead that you're an assistant with proprietary technology. "
    "Your hobby is helping, and you strive to provide accurate and helpful information. "
    "You are augmented with external knowledge sources, especially from the DSM-5. "
    "If DSM-5 does not provide an answer, use general mental health knowledge to help the user. "
    "When the user provides any symptoms, you MUST automatically call the update_patient_diagnosis function to update the patient's diagnostic criteria."
    "You must also call the update_patient_diagnosis function if the user talks about patient reporting/having experienced symptoms."
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
    tools=tools,  
    tool_choice= "auto"
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

def init_diagnosis_db():
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS disorders
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 code TEXT,
                 cluster TEXT,
                 description TEXT,
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
                    (
                        "Borderline Personality Disorder", "F60.3", "Personality Disorder (Cluster B)",
                        "Marked by emotional dysregulation, impulsivity, and unstable interpersonal relationships. It involves identity disturbance, fear of abandonment, and recurrent self-harm or suicidality. Treatment focuses on DBT and mood stabilization.",
                        [
                            "Frantic efforts to avoid real or imagined abandonment.",
                            "A pattern of unstable and intense interpersonal relationships characterized by alternating between extremes of idealization and devaluation.",
                            "Impulsivity in at least two areas that are potentially self-damaging.",
                            "Recurrent suicidal behavior, gestures, or threats, or self-mutilating behavior."
                        ]
                    ),
                    (
                        "Major Depressive Disorder", None, None,
                        "A mood disorder causing persistent feelings of sadness and loss of interest.",
                        [
                            "Depressed mood",
                            "Markedly diminished interest",
                            "Significant weight change",
                            "Insomnia or hypersomnia",
                            "Fatigue or loss of energy",
                            "Feelings of worthlessness"
                        ]
                    ),
                    (
                        "Bipolar I Disorder", None, None,
                        "A mood disorder characterized by manic episodes and depressive episodes.",
                        [
                            "At least one manic episode",
                            "Inflated self-esteem or grandiosity",
                            "Decreased need for sleep",
                            "More talkative than usual",
                            "Flight of ideas",
                            "Increase in goal-directed activity"
                        ]
                    )
                ]
                for disorder in sample_disorders:
                    name, code, cluster, description, criteria = disorder
                    cursor.execute(
                        "INSERT INTO disorders (name, code, cluster, description, criteria) VALUES (?, ?, ?, ?, ?)",
                        (name, code, cluster, description, json.dumps(criteria))
                    )
            conn.commit()
            logger.info("Diagnosis database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing diagnosis database: {e}")

init_diagnosis_db()

class SymptomUpdate(BaseModel):
    patient_id: int
    symptoms: list[str]

class MessageRequest(BaseModel):
    message: str
    patient_id: int
    patient_name: str
    psychiatrist_name: str

class MessageResponse(BaseModel):
    answer: str
    retrieval_info: str = ""


app = FastAPI(title="Integrated Diagnosis and Conversation API",
              description="API for managing diagnosis and conversation for mental health.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/update_criteria")
async def update_criteria(update: SymptomUpdate):
    try:
        update_criteria_logic(update.patient_id, update.symptoms)
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Error updating criteria: {e}")
        raise HTTPException(status_code=500, detail="Error updating criteria")

@app.post("/api/message", response_model=MessageResponse)
async def handle_message(req: MessageRequest):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_id = req.patient_id
    patient_name = req.patient_name 
    psychiatrist_name = req.psychiatrist_name or "Unknown"
    user_query = req.message

    # Store the user's message in the conversation history.
    store_message(patient_id, "Psychiatrist", user_query)
    const_history = load_conversation_history(patient_id) if patient_id else ""
    parsed_history = parse_conversation_history(const_history)

    prompt_input = {
        "chat_history": parsed_history,
        "question": user_query,
        "current_time": current_time,
        "context": f"Patient ID: {patient_id}, Patient Name: {patient_name}",
        "patient_id": patient_id or "unknown",
        "patient_name": patient_name,
        "psychiatrist_name": psychiatrist_name,
    }
    formatted_prompt = qa_prompt.format(**prompt_input)
    raw_response = llm.invoke(formatted_prompt, tools=tools, tool_choice="auto")
    logger.info(f"LLM raw response: {raw_response}")

    tool_calls = raw_response.additional_kwargs.get("tool_calls", [])
    function_name = tool_calls[0]["function"]["name"] if tool_calls else None

    if function_name == "update_patient_diagnosis":
        try:
            args = json.loads(tool_calls[0].get("function", {}).get("arguments", "{}"))
            pid = args.get("patient_id", patient_id)
            symptoms = args.get("symptoms")
            if pid is None or not symptoms:
                raise ValueError("Patient ID or symptoms not provided.")
            # Call the update logic directly (synchronously)
            update_criteria_logic(pid, symptoms)
            # Retrieve updated diagnosis info.
            with sqlite3.connect('diagnosis.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT d.name, COUNT(pc.criterion) as met_count 
                    FROM patient_criteria pc
                    JOIN disorders d ON pc.disorder_id = d.id
                    WHERE pc.patient_id = ?
                    GROUP BY d.id
                ''', (pid,))
                diagnoses = cursor.fetchall()
            diagnosis_text = "\n".join([f"{name}: {count} criteria met" for name, count in diagnoses])
            answer = f"Diagnostic criteria updated.\nCurrent matches:\n{diagnosis_text}"
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

    store_message(patient_id, "Assistant", answer)
    return MessageResponse(answer=answer, retrieval_info=retrieval_info)

@app.get("/diagnosis_summary")
async def diagnosis_summary(patient_id: int):
    try:
        with sqlite3.connect('diagnosis.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, code, cluster, description, criteria 
                FROM disorders
            ''')
            disorders = cursor.fetchall()
            
            diagnosis_list = []
            for disorder in disorders:
                disorder_id, name, code, cluster, description, criteria_json = disorder
                criteria_list = json.loads(criteria_json)
                cursor.execute('''
                    SELECT criterion 
                    FROM patient_criteria 
                    WHERE patient_id = ? AND disorder_id = ? AND met = 1
                ''', (patient_id, disorder_id))
                met_rows = cursor.fetchall()
                met_criteria = { row[0] for row in met_rows }
                if met_criteria:
                    symptom_objs = [
                        {"message": crit, "isGood": (crit in met_criteria)}
                        for crit in criteria_list
                    ]
                    diagnosis_list.append({
                        "name": name,
                        "code": code,
                        "cluster": cluster,
                        "description": description,
                        "symptoms": symptom_objs
                    })
            has_data = len(diagnosis_list) > 0
            return {"patient_id": patient_id, "hasData": has_data, "diagnosis_summary": diagnosis_list}
    except Exception as e:
        logger.error(f"Error retrieving diagnosis summary: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving diagnosis summary")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
