# Smart Agri AI - Upgraded Version

## What is upgraded
- Professional FastAPI backend
- SQLite history storage
- Health, latest, and history APIs
- Better ML pipeline with confidence score
- Better fertilizer + irrigation + soil analysis
- Improved ESP32 firmware loop
- Live dashboard UI with chart and history

## Files
- `main.py` -> Backend API
- `model.py` -> ML + recommendation engine
- `firmware.py` -> ESP32 MicroPython firmware
- `index.html` -> Dashboard frontend
- `requirements.txt` -> Python dependencies

## Run backend
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Open frontend
Just open `index.html` in your browser.

## Important notes
- Put `Crop_recommendation.csv` inside `data/` or `../data/`
- Update `LAPTOP_IP` in `firmware.py` when your local IP changes
- P and K are still placeholders until you connect actual sensors
