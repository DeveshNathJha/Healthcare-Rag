#!/bin/bash
echo "Creating virtual environment: rag-health..."
python3 -m venv rag-health

echo "Activating environment..."
source rag-health/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. To start the server later, run:"
echo "source rag-health/bin/activate && uvicorn app.main:app --reload"