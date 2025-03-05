from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import uvicorn

# Definier een vaste input structuur voor de API
class FixtureInput(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    venue: int
    city: int
    round: int
    league: int
    home_team: int
    away_team: int
    referee: int

# Maak een FastAPI app instance
app = FastAPI()

# Laad het getrainde model
model = joblib.load('dump/best_rf_model.pkl')

# Definieer een API endpoint voor het maken van voorspellingen
@app.post("/predict")
async def predict(fixture: List[FixtureInput]):
    try:
        # Convert incoming data to DataFrame
        data = [item.dict() for item in fixture]
        df_input = pd.DataFrame(data)
        
        predictions = model.predict(df_input)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    # Run the API server
    uvicorn.run(app, host='127.0.0.1', port=8000)