## mindseek RAG

Contains files needed for the RAG setup.

Steps to run:
1. Insert OpenAPI Key in your .env 
3. Run `python extractor.py`
4. Run `python embedder.py`
5. Run `main.py`
6. Run `uvicorn main:app --reload`
7. Run `streamlit run app.py`

Make sure you have all the dependencies required.
