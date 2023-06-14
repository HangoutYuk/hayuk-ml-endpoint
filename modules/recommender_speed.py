from fastapi import HTTPException
from modules.recommender import *
import pandas as pd
import haversine as hs
import requests
import mysql.connector
import time


# Define - Global Variables
MAX_ITEM = 10


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


def refresh_sentiment_data():
    connection = connect_database()
    cursor = connection.cursor()
    sql = "UPDATE location_speed SET quality = %s WHERE place_id = %s"
    val = []
    place_id, _, _, _, text_review, _ = get_data_mysql_speed(connection)

    prediction_results = []
    for i in text_review:
        i2 = preprocess(i)
        prediction = request_endpoints(i2)  # You Were To Predict Here
        prediction_results.append(prediction)

    for i, j in zip(place_id, prediction):
        val.append(tuple([j, i]))

    cursor.executemany(sql, val)
    connection.commit()
    return {
        "message": "Data Refreshed",
        "detail": f"{cursor.rowcount} record updated successfully"
    }


def get_data_mysql_speed(conn):
    place_id = []
    name = []
    latitude = []
    longitude = []
    text_review = []
    quality = []

    conn = conn.cursor()
    conn.execute("SELECT * FROM location_speed")

    myresult = conn.fetchall()

    for x in myresult:
        place_id.append(x[0])
        name.append(x[1])
        latitude.append(x[2])
        longitude.append(x[3])
        text_review.append(x[4])
        quality.append(x[5])

    return place_id, name, latitude, longitude, text_review, quality


def recommender_place_speed(user_location: tuple):
    # start time
    st = time.time()

    # grab mysql connection database
    connection = connect_database()

    # fetch all necessary items
    place_id, name, latitude, longitude, _, quality = get_data_mysql_speed(
        connection)

    # convert data to pandas dataframe
    df_temp = {
        'place_id': place_id,
        'name': name,
        'latitude': latitude,
        'longitude': longitude,
        'quality': quality
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
    ENDPOINT_ID = "7950806074959855616"
    PROJECT_ID = "curious-furnace-381420"
    access_token = "ya29.a0AWY7CkmZIUi034IGsGUyg7p_-SEdf8SVnfTZFE6CD9l-qAFadGGv8RoQsER8DTSNwxZQtRSXJLGiXUZuNM4RNRnji79cKFbO86zKlx78KY2mIvXlI9kVVfyuZgl2iKw-4GqyGF3gFxfgENPplpmdp4wo2Ra1A14R6KOnUAKvKn6mnOJTkvVAJhWx4ewuAd4RPut-J_lvKESj5QnB2H6jXoQocOqR0N3vviGVVocaCgYKAdwSARMSFQG1tDrpUn3da4Xozbyw6iHcdlrxHw0238"
    req = requests.post(f"https://asia-southeast1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/asia-southeast1/endpoints/{ENDPOINT_ID}:predict", headers={
                        'Content-Type': 'application/json', 'Authorization': f'Bearer {access_token}'}, json={"instances": [[sentence]]})
    return req.text
