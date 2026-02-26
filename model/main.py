from model.text_to_food import text_to_food
from fastapi import FastAPI

app = FastAPI()

model_path = r'model/results/checkpoint-1125'

@app.get("/predict")
def predict(text: str):
    return text_to_food(text, model_path)