from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import re
app = FastAPI()

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('col.pickle', 'rb') as f:
    col = pickle.load(f)


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


def parse_date(item):
    arr = [0 for i in range(len(col))]
    fuel = 'fuel_' + item.fuel
    seller_type = 'seller_type_' + item.seller_type
    owner = 'owner_' + item.owner
    seats = 'seats_' + str(int(item.seats))

    for i in range(len(col)):
        if col[i] == 'year' or col[i] == 'km_driven':
            arr[i] = int(getattr(item, col[i]))

        if col[i] == 'mileage':
            match = re.search(r'\d+\.\d+', str(getattr(item, col[i])))
            arr[i] = float(match.group()) if match else 0

        if col[i] == 'engine':
            match = re.search(r'\d+', str(getattr(item, col[i])))
            arr[i] = int(match.group()) if match else 0

        if col[i] == 'max_power':
            match = re.search(r'\d+\.*\d*', str(getattr(item, col[i])))
            arr[i] = float(match.group()) if match else 0
        if col[i] == seats:
            arr[i] = 1
        if col[i] == fuel:
            arr[i] = 1
        if col[i] == seller_type:
            arr[i] = 1
        if col[i] == owner:
            arr[i] = 1

    data = pd.DataFrame(columns=col)
    data.loc[len(data)] = arr
    data_subset = data.drop(['max_power', 'mileage'], axis=1)

    for column in data_subset.columns:
        data[column] = data[column].astype(int)
    prediction = model.predict(data)
    return prediction


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return parse_date(item)


@app.post("/predict_items")
def predict_items(items: List[Item]):
    predictions = []
    data = {item_data: [] for item_data in list(items[0].model_dump().keys())}
    сols = data.keys()

    for item in items:
        prediction_item = parse_date(item)
        predictions.append(float(prediction_item))

        for key in сols:
            data[key].append(item.model_dump()[key])

    data['predict'] = predictions
    return data
