import pandas as pd
import haversine as hs
import tensorflow as tf
import os
from fastapi import HTTPException

# Define - Global Variables
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATABASE_PATH_LOCATION = os.path.join(
    CURRENT_PATH, "data", "location.csv")
MODEL_PATH = os.path.join(CURRENT_PATH, "models")
MAX_ITEM = 10


def pengubah_lat_long(lokasi):
    try:
        index = lokasi.find(",")
        lat = float(lokasi[:index].strip())
        long = float(lokasi[index+1:].strip())
        return (lat, long)
    except:
        raise HTTPException(status_code=400, detail="Bad Request")


def distance_from(loc1, loc2):
    distance = hs.haversine(loc1, loc2)
    return distance


def to_list_of_dict(data):
    list_of_dict = []
    for index, value in enumerate(data):
        list_of_dict.append({index: value})
    return list_of_dict


def recommender_place(user_location: tuple):

    # mengambil data
    df = pd.read_csv(DATABASE_PATH_LOCATION)

    # mempersempit tabel
    df_simplified = df[
        ['place_id', 'name', 'latitude', 'longitude', 'text_review']]

    # membuat kolom coor yang berisi nilai latitude dan longitude
    df_simplified = df_simplified.assign(coor=list(
        zip(df_simplified.latitude, df_simplified.longitude)))

    # melakukan pengulangan untuk mendapatkan nilai jarak
    distances_km = []
    for row in df_simplified.itertuples(index=False):
        distances_km.append(distance_from(user_location, row[5]))

    # memasukkan nilai jarak ke kolom distance_from_user
    df_simplified = df_simplified.assign(distance_from_user=distances_km)

    # menginisiasi model
    model = tf.saved_model.load(MODEL_PATH)

    # melakukan prediksi sentimen
    sentiment = []
    for sentence in df_simplified['text_review']:
        if len(str(sentence)) <= 1:
            sentiment.append(0.0)
        else:
            prediction = model([[sentence]]).numpy()[0][0]
            sentiment.append(prediction)

    # memasukkan hasil prediksi ke dalam kolom quality
    df_simplified = df_simplified.assign(quality=sentiment)

    # mengurutkan data berdasarkan jarak terpendek dan prediksi sentimen
    df_oversimplified = df_simplified.sort_values(
        ['distance_from_user', 'quality'], ascending=[True, True]).head(MAX_ITEM)

    # mengembalikan nilai respons
    ids = to_list_of_dict(df_oversimplified.loc[:, "place_id"])
    names = to_list_of_dict(df_oversimplified.loc[:, "name"])

    return {"places_id": ids, "places_name": names}
