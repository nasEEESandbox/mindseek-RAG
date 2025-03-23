import os
import json
import sqlite3
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
SIMILARITY_THRESHOLD = 0.5 # Minimum similarity for a symptom to be considered a criterion, adjust as needed

def init_db():
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
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

init_db()

app = FastAPI(title="Diagnosis API", description="API for diagnosis and disorder management.")

# Pydantic models
class SymptomUpdate(BaseModel):
    patient_id: int
    symptoms: list[str]

class PatientCreate(BaseModel):
    name: str

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
                    # Use embed_query instead of embed
                    crit_emb = np.array(embedding_function.embed_query(criterion))
                    criterion_embeds.append((criterion, crit_emb))
                disorder_criteria_embeddings[disorder_id] = criterion_embeds
                logger.info(f"Precomputed embeddings for disorder {disorder_id}")
            
            # Process each user symptom
            for symptom in update.symptoms:
                symptom_emb = np.array(embedding_function.embed_query(symptom))
                for disorder_id, criterion_list in disorder_criteria_embeddings.items():
                    for criterion, crit_emb in criterion_list:
                        norm_symptom = np.linalg.norm(symptom_emb)
                        norm_crit = np.linalg.norm(crit_emb)
                        if norm_symptom == 0 or norm_crit == 0:
                            continue
                        sim = np.dot(symptom_emb, crit_emb) / (norm_symptom * norm_crit)
                        logger.info(f"Similarity between '{symptom}' and '{criterion}': {sim}")
                        if sim >= SIMILARITY_THRESHOLD:
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
    """
    Retrieve a summary of all criteria met for the patient along with disorder names.
    """
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
            
            # Group criteria by disorder name
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