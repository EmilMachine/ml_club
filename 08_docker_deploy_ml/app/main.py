from datetime import datetime
import logging
import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

import numpy as np
import pandas as pd
import xgboost
import joblib
from pydantic import BaseModel


import uvicorn

logger = logging.getLogger("model_titanic_survival")

v1_path = "/v1"
model_path = "/titanic"

#/v1/titanic/single/{country}

logger.info("Loading API")
app = FastAPI(
    title="Titanic survival",
    description="This API exposes predictions for survival of the Titanic disaster.",
    version="0.1.0",
    openapi_url=f"{v1_path}{model_path}/openapi.json",
    docs_url=f"{v1_path}{model_path}/docs",
    redoc_url=f"{v1_path}{model_path}/redoc",
)


predictions_path = os.getenv("PREDICTIONS_PATH", "data/model_xgboost_best.sav")
logger.info(f"Loading predictions from {predictions_path}")
loaded_model = joblib.load(predictions_path)


def get_prediction(x):
    y = loaded_model.predict(x)[0]  # just get single value
    prob = loaded_model.predict_proba(x)[0].tolist()  # send to list for return
    return {'prediction': int(y), 'probability': prob}


class ModelParams(BaseModel):
    PassengerId   : np.int
    Pclass        : np.int
    Sex           : np.int
    Age           : np.float
    SibSp         : np.int
    Parch         : np.int
    Fare          : np.float
    FamilySize    : np.int
    Has_Cabin     : np.int
    Name_length   : np.int
    Ticket_length : np.int
    Embarked_Q    : np.int
    Embarked_S    : np.int
    Embarked_C    : np.int


@app.post(f"{v1_path}{model_path}/predict")
def predict(params: ModelParams):

    #pd.Dataframe([model.dict() for model in data])

    df = pd.DataFrame(params.dict(), index=[1])
    pred = get_prediction(df)
    return pred

# Official documentation for fastapi
# https://fastapi.tiangolo.com/tutorial/path-params/


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
