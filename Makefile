.PHONY: install clean run demo ui

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .faiss_cache/

demo:
	python demo.py

run:
	python rag_pipeline.py

# Lance l'interface Streamlit depuis le venv (obligatoire pour pdfminer + numpy<2)
ui:
	.venv\Scripts\streamlit.exe run app.py
