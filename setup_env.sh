#!/bin/bash
echo "Creating virtual environment: rag-dev..."
python3 -m venv rag-dev

echo "Activating environment..."
source rag-dev/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. To start the server later, run:"
echo "source rag-dev/bin/activate && uvicorn app.main:app --reload"