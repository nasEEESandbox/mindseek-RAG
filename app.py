import os
import re
import sqlite3
import json
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import requests
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate

# -----------------------------------------
# Define the update_patient_diagnosis tool for function calling.
# -----------------------------------------
tools = [{
    "name": "update_patient_diagnosis",
    "description": "Update patient's diagnostic criteria based on reported symptoms. Requires patient_id and symptoms. This function must be called automatically if the user provides any symptoms.",
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
# Load environment variables
# -----------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not set! Please set it in your environment or .env file.")
    st.stop()

# -----------------------------------------
# Initialize vector database (Chroma)
# -----------------------------------------
persist_directory = "./chroma_db"
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# -----------------------------------------
# Function to fetch patient name from database given patient_id
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
# System message for chatbot behavior (with stronger instructions for function calling)
# -----------------------------------------
system_msg = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant called MindSeek that remembers details from the conversation. "
    "The user is a psychiatrist, not the patient. Always refer to the patient using the provided patient details. "
    "You must not reveal that you're an OpenAI model or GPT; say instead that you're an assistant with proprietary technology. "
    "Your hobby is helping, and you strive to provide accurate and helpful information. "
    "You are augmented with external knowledge sources, especially from the DSM-5. "
    "If DSM-5 does not provide an answer, use general mental health knowledge to help the user. "
    "You can provide information on mental health conditions, symptoms, treatments, and general advice on well-being. "
    "Do not respond to queries unrelated to mental health or DSM-5 disorders unless explicitly asked. "
    "When the user provides any symptoms, you MUST automatically call the update_patient_diagnosis function to update the patient's diagnostic criteria. "
    "Also, if the user asks about the patient's name, provide it. "
    "Remember: The user is a psychiatrist seeking assistance with their patient's case."
)

# -----------------------------------------
# Human message prompt including current time, patient context, and psychiatrist name.
# -----------------------------------------
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

# -----------------------------------------
# Initialize LLM (using ChatOpenAI) with function calling enabled.
# -----------------------------------------
llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0.7,
    functions=tools,            # Provide our update_patient_diagnosis tool.
    function_call="auto"         # Let the model decide automatically.
)

# -----------------------------------------
# Create a question generator chain to rephrase follow-up questions.
# -----------------------------------------
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

# -----------------------------------------
# Ensure memory persists across user interactions.
# -----------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question",
        return_messages=True, 
        output_key="answer"
    )

# -----------------------------------------
# Load the QA chain using the custom prompt.
# -----------------------------------------
qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

# -----------------------------------------
# Create Conversational Retrieval Chain with the question generator.
# -----------------------------------------
conversational_chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator_chain,
    combine_docs_chain=qa_chain,
    memory=st.session_state.memory,
    return_source_documents=True
)

# -----------------------------------------
# Query Relevance Check using a simple prompt.
# -----------------------------------------
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
# Streamlit UI Setup.
# -----------------------------------------
st.title("💬 DSM-5 RAG Chatbot")

# Sidebar for patient ID input.
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

patient_id_input = st.sidebar.text_input("Enter Patient ID (or leave blank if unknown):")
if patient_id_input:
    try:
        st.session_state.patient_id = int(patient_id_input)
    except ValueError:
        st.sidebar.error("Patient ID must be an integer.")

# Sidebar for psychiatrist name input.
if "psychiatrist_name" not in st.session_state:
    st.session_state.psychiatrist_name = ""
psychiatrist_name_input = st.sidebar.text_input("Enter your name (Psychiatrist):")
if psychiatrist_name_input:
    st.session_state.psychiatrist_name = psychiatrist_name_input

# Display the current patient ID, patient name, and psychiatrist name in the sidebar if available.
if st.session_state.patient_id is not None:
    patient_name = get_patient_name(st.session_state.patient_id)
    st.sidebar.write(f"Current Patient ID: **{st.session_state.patient_id}**")
    st.sidebar.write(f"Patient Name: **{patient_name}**")
else:
    st.sidebar.write("No Patient ID provided.")
if st.session_state.psychiatrist_name:
    st.sidebar.write(f"Psychiatrist: **{st.session_state.psychiatrist_name}**")
else:
    st.sidebar.write("No Psychiatrist name provided.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Type your message:")

if query:
    with st.spinner("Thinking..."):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = st.session_state.patient_id
        patient_name = get_patient_name(patient_id) if patient_id is not None else "Unknown"
        psychiatrist_name = st.session_state.psychiatrist_name or "Unknown"
        
        # Let the LLM decide automatically via function calling.
        raw_response = llm.invoke(query)
        function_call = raw_response.additional_kwargs.get("function_call")
        
        if function_call and function_call.get("name") == "update_patient_diagnosis":
            try:
                args = json.loads(function_call.get("arguments", "{}"))
                # Use the patient ID from the function call if available, otherwise fallback to session state.
                patient_id = args.get("patient_id") or st.session_state.patient_id
                if patient_id is None:
                    raise ValueError("Patient ID is not provided.")
                response = requests.post(
                    "http://localhost:8000/update_criteria",
                    json={"patient_id": patient_id, "symptoms": args["symptoms"]}
                )
                
                # Get potential diagnoses from local database for display.
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
            # Process as a mental health query.
            relevance_result = relevance_chain.invoke({"question": query})
            is_relevant = relevance_result["text"].strip().lower()
            if is_relevant == "yes":
                # Inject patient and psychiatrist context into the conversation.
                context = f"Patient ID: {patient_id}, Patient Name: {patient_name}" if patient_id is not None else ""
                result = conversational_chain({
                    "question": query,
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
                fallback_chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", "")
                answer = qa_chain.run({
                    "chat_history": fallback_chat_history,
                    "patient_id": patient_id,      
                    "patient_name": patient_name,  
                    "psychiatrist_name": psychiatrist_name,
                    "context": "",
                    "question": query,
                    "current_time": current_time,
                    "input_documents": [],
                })
                retrieval_info = "No retrieval was performed."
        
        st.session_state.memory.save_context({"question": query}, {"answer": answer})
        st.session_state.chat_history.append(("🧑 You", query))
        st.session_state.chat_history.append(("🤖 Chatbot", answer + "\n\n" + retrieval_info))

for speaker, message in st.session_state.chat_history:
    st.write(f"**{speaker}:** {message}")

st.write("**Debug: Full Conversation Memory:**")
st.write(st.session_state.memory.load_memory_variables({})["chat_history"])
