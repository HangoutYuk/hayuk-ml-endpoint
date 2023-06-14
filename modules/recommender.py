from fastapi import HTTPException
from cleantext import clean
from dotenv import load_dotenv
import os
import pandas as pd
import haversine as hs
import requests
import mysql.connector
import time
import re

load_dotenv()

# Define - Global Variables
MAX_ITEM = 10


def preprocess(sentence):
    sentence1 = clean(sentence, no_emoji=True)
    sentence2 = sentence1.lower().strip()
    words = sentence2.split()
    temp = []
    for i in words:
        normal_string = re.sub("[^A-Z]", "", i, 0, re.IGNORECASE)
        temp.append(normal_string)
    sentence3 = " ".join(temp)

    return sentence3


def pengubah_lat_long(lokasi):
    try:
        index = lokasi.find(",")
        lat = float(lokasi[:index].strip())
        long = float(lokasi[index+1:].strip())
        return (lat, long)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Bad Request : {e}")


def distance_from(loc1, loc2):
    distance = hs.haversine(loc1, loc2)
    return distance


def connect_database():
    try:
        mydb = mysql.connector.connect(
            host="34.101.95.234",
            user="root",
            password="9gYnTWLACG3dkhA",
            database="hayuk_ml"
        )
        return mydb
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Server Connect Fail : {e}")


def get_data_mysql(conn):
    place_id = []
    name = []
    latitude = []
    longitude = []
    text_review = []

    conn = conn.cursor()
    conn.execute("SELECT * FROM location")

    myresult = conn.fetchall()

    for x in myresult:
        place_id.append(x[0])
        name.append(x[1])
        latitude.append(x[2])
        longitude.append(x[3])
        text_review.append(x[4])

    return place_id, name, latitude, longitude, text_review


def to_list_of_dict(data):
    list_of_dict = []
    for index, value in enumerate(data):
        list_of_dict.append({index: value})
    return list_of_dict


def recommender_place(user_location: tuple):
    # start time
    st = time.time()

    # grab mysql connection database
    connection = connect_database()

    # fetch all necessary items
    place_id, name, latitude, longitude, text_review = get_data_mysql(
        connection)

    # convert data to pandas dataframe
    df_temp = {
        'place_id': place_id,
        'name': name,
        'latitude': latitude,
        'longitude': longitude,
        'text_review': text_review
    }

    # mempersempit tabel
    df = pd.DataFrame(df_temp)

    # membuat kolom coor yang berisi nilai latitude dan longitude
    df = df.assign(coor=list(zip(df.latitude, df.longitude)))

    # melakukan pengulangan untuk mendapatkan nilai jarak
    distances_km = []
    for row in df.itertuples(index=False):
        distances_km.append(distance_from(user_location, row[5]))

    # memasukkan nilai jarak ke kolom distance_from_user
    df = df.assign(distance_from_user=distances_km)

    df_distance_selected = df.sort_values(
        ['distance_from_user'], ascending=[True]).head(MAX_ITEM*2)

    # melakukan prediksi sentimen
    sentiment = []
    for sentence in df_distance_selected['text_review']:
        sentence1 = preprocess(sentence)
        prediction = request_endpoints(sentence1)
        sentiment.append(prediction)

    # memasukkan hasil prediksi ke dalam kolom quality
    df_distance_selected = df_distance_selected.assign(quality=sentiment)

    # mengurutkan data berdasarkan jarak terpendek dan prediksi sentimen
    df_quality_selected = df_distance_selected.sort_values(
        ['quality'], ascending=[True]).head(MAX_ITEM)

    # end time
    et = time.time()

    # calculate elapsed time
    elapsed_time = et - st

    # wrap results into variable
    ids = to_list_of_dict(df_quality_selected.loc[:, "place_id"])
    names = to_list_of_dict(df_quality_selected.loc[:, "name"])

    return {"places_id": ids, "places_name": names, "elapsed_time": elapsed_time}


def request_endpoints(sentence: str):
    ENDPOINT_ID = os.environ.get("ENDPOINT_ID")
    PROJECT_ID = os.environ.get("PROJECT_ID")
    access_token = os.environ.get("ACCESS_TOKEN")
    req = requests.post(f"https://asia-southeast1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/asia-southeast1/endpoints/{ENDPOINT_ID}:predict", headers={
                        'Content-Type': 'application/json', 'Authorization': f'Bearer {access_token}'}, json={"instances": [[sentence]]})
    return req.text
