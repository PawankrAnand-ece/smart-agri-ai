from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import get_recommendation
import sqlite3

app = FastAPI(title="AI Smart Agriculture System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = "agri_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        N INTEGER,
        P INTEGER,
        K INTEGER,
        moisture INTEGER,
        temperature REAL,
        crop TEXT,
        fertilizer TEXT,
        price INTEGER
    )
    """)

    conn.commit()
    conn.close()

init_db()

class SensorData(BaseModel):
    N: int
    P: int
    K: int
    moisture: int
    temperature: float

def save_to_db(sensor, result):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO records (N, P, K, moisture, temperature, crop, fertilizer, price)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sensor["N"],
        sensor["P"],
        sensor["K"],
        sensor["moisture"],
        sensor["temperature"],
        result["crop"],
        result["fertilizer"],
        result["market"]["price_per_quintal"]
    ))

    conn.commit()
    conn.close()

@app.get("/")
def home():
    return {"message": "AI Smart Agriculture API is LIVE!"}

@app.post("/api/data")
async def predict(data: SensorData):
    sensor_dict = data.dict()
    print("Received from ESP32:", sensor_dict)

    result = get_recommendation(sensor_dict)
    save_to_db(sensor_dict, result)

    print("Recommendation sent:", result)
    return result

@app.get("/api/history")
def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, timestamp, N, P, K, moisture, temperature, crop, fertilizer, price
    FROM records
    ORDER BY id DESC
    LIMIT 25
    """)

    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "N": row[2],
            "P": row[3],
            "K": row[4],
            "moisture": row[5],
            "temperature": row[6],
            "crop": row[7],
            "fertilizer": row[8],
            "price": row[9]
        })

    return history

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)