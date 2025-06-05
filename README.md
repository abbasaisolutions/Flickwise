# FlickWise

A hybrid recommendation engine using TensorFlow/Keras, Scikit-learn, FastAPI, and Streamlit.

## Project Structure

- `backend/`: FastAPI backend (model loading and prediction)
- `frontend/`: Streamlit dashboard UI
- `rootfiles/`: Saved ML model and preprocessing artifacts

## Getting Started

```bash
# Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
streamlit run frontend/dashboard.py
