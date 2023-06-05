from fastapi import HTTPException
import pandas as pd
import haversine as hs
import os
import requests
import mysql.connector

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
    except Exception as e:
        raise HTTPException(status_code=400, detail="Bad Request",
                            descriptions="Input parameters should be (latitude,longitude) without brackets", messages=e)


def distance_from(loc1, loc2):
    distance = hs.haversine(loc1, loc2)
    return distance


def connect_database():
    try:
        mydb = mysql.connector.connect(
            host="34.101.154.108",
            user="root",
            password="9gYnTWLACG3dkhA",
            database="hayuk_ml"
        )
        return mydb
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Server Connect Fail", messages=e)


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
    df = df.assign(coor=list(
        zip(df.latitude, df.longitude)))

    # melakukan pengulangan untuk mendapatkan nilai jarak
    distances_km = []
    for row in df.itertuples(index=False):
        distances_km.append(distance_from(user_location, row[5]))

    # memasukkan nilai jarak ke kolom distance_from_user
    df = df.assign(distance_from_user=distances_km)

    df_distance_selected = df.sort_values(
        ['distance_from_user'], ascending=[True]).head(MAX_ITEM+10)

    # menginisiasi model
    # model = tf.saved_model.load(MODEL_PATH)


    # melakukan prediksi sentimen
    sentiment = []
    for sentence in df_distance_selected['text_review']:
        if len(str(sentence)) <= 1:
            sentiment.append(0.0)
        else:
            prediction = request_endpoints(sentence)
            sentiment.append(prediction)

    # memasukkan hasil prediksi ke dalam kolom quality
    df_distance_selected = df_distance_selected.assign(quality=sentiment)

    # mengurutkan data berdasarkan jarak terpendek dan prediksi sentimen
    df_quality_selected = df_distance_selected.sort_values(
        ['quality'], ascending=[True]).head(MAX_ITEM)

    ids = to_list_of_dict(df_quality_selected.loc[:, "place_id"])
    names = to_list_of_dict(df_quality_selected.loc[:, "name"])

    return {"places_id": ids, "places_name": names}


def request_endpoints(sentence: str):
    ENDPOINT_ID="7950806074959855616"
    PROJECT_ID="curious-furnace-381420"
    access_token = "ya29.a0AWY7CkmZIUi034IGsGUyg7p_-SEdf8SVnfTZFE6CD9l-qAFadGGv8RoQsER8DTSNwxZQtRSXJLGiXUZuNM4RNRnji79cKFbO86zKlx78KY2mIvXlI9kVVfyuZgl2iKw-4GqyGF3gFxfgENPplpmdp4wo2Ra1A14R6KOnUAKvKn6mnOJTkvVAJhWx4ewuAd4RPut-J_lvKESj5QnB2H6jXoQocOqR0N3vviGVVocaCgYKAdwSARMSFQG1tDrpUn3da4Xozbyw6iHcdlrxHw0238"
    req = requests.post("https://asia-southeast1-aiplatform.googleapis.com/v1/projects/{}/locations/asia-southeast1/endpoints/{}:predict".format(PROJECT_ID, ENDPOINT_ID), headers={'Content-Type':'application/json',
               'Authorization': 'Bearer {}'.format(access_token)}, json={"instances": [[sentence]]})
    print(req.text)
    return req.text
