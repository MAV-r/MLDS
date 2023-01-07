from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from fastapi.encoders import jsonable_encoder
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

with open("linear_regression.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

#scaler.transform(X_test)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    numeric_col = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    df_test=pd.DataFrame(jsonable_encoder(item), index=[0])
    df_test = df_test.drop(["torque"], axis=1)
    df_test['engine']=df_test['engine'].str[:-3].str.strip()
    df_test['max_power'] = df_test['max_power'].str[:-4].str.strip()
    df_test['engine'] = df_test['engine'].astype(float)
    df_test['max_power'] = df_test['max_power'].astype(float)
    df_test['mileage_clear'] = np.where(df_test['mileage'].str[-5:] == "km/kg", "km/kg", "kmpl")
    df_test['mileage'] = df_test['mileage'].str[:-5].str.strip()
    df_test['mileage'] = df_test['mileage'].astype(float)
    df_test['mileage'] = np.where(df_test['mileage_clear'] == "km/kg", df_test['mileage'] * 9.8, df_test['mileage'])
    df_test.drop("mileage_clear", axis=1, inplace=True)
    df_test['engine'] = df_test['engine'].astype(int)
    df_test['seats'] = df_test['seats'].astype(int)
    X_test  = df_test[numeric_col].drop('selling_price', axis=1)
    t=scaler.transform(X_test)
    return model.predict(t)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    numeric_col = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    df_test=pd.DataFrame(jsonable_encoder(items))
    df_test = df_test.drop(["torque"], axis=1)
    df_test['engine']=df_test['engine'].str[:-3].str.strip()
    df_test['max_power'] = df_test['max_power'].str[:-4].str.strip()
    df_test['engine'] = df_test['engine'].astype(float)
    df_test['max_power'] = df_test['max_power'].astype(float)
    df_test['mileage_clear'] = np.where(df_test['mileage'].str[-5:] == "km/kg", "km/kg", "kmpl")
    df_test['mileage'] = df_test['mileage'].str[:-5].str.strip()
    df_test['mileage'] = df_test['mileage'].astype(float)
    df_test['mileage'] = np.where(df_test['mileage_clear'] == "km/kg", df_test['mileage'] * 9.8, df_test['mileage'])
    df_test.drop("mileage_clear", axis=1, inplace=True)
    df_test['engine'] = df_test['engine'].astype(int)
    df_test['seats'] = df_test['seats'].astype(int)
    X_test  = df_test[numeric_col].drop('selling_price', axis=1)
    t=scaler.transform(X_test)
    ans=pd.DataFrame(model.predict(t))
    ans.columns=['predicted_price']

    return ans
