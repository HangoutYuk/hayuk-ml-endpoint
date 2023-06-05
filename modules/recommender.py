from fastapi import HTTPException
import pandas as pd
import haversine as hs
import tensorflow as tf
import os
import mysql.connector
import time

# Define - Global Variables
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATABASE_PATH_LOCATION = os.path.join(
    CURRENT_PATH, "data", "loc.csv")
MODEL_PATH = os.path.join(CURRENT_PATH, "models")
MAX_ITEM = 5
ADD_ITEM = 5


def pengubah_lat_long(lokasi):
    try:
        index = lokasi.find(",")
        lat = float(lokasi[:index].strip())
        long = float(lokasi[index+1:].strip())
        return (lat, long)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad Request : {e}")


def distance_from(loc1, loc2):
    distance = hs.haversine(loc1, loc2)
    return distance


def connect_database():
    try:
        mydb = mysql.connector.connect(
            # host="34.121.171.7",
            # user="root",
            # password="9gYnTWLACG3dkhA",
            # database="hayuk_ml"
            host="localhost",
            user="root",
            password="",
            database="hayuk_ml"
        )
        return mydb
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Server Connect Fail : {e}")


def get_data_mysql(conn):
    try:
        place_id = []
        name = []
        latitude = []
        longitude = []
        text_review = []
        quality = []

        conn = conn.cursor()
        conn.execute("SELECT * FROM location")

        myresult = conn.fetchall()

        for x in myresult:
            place_id.append(x[0])
            name.append(x[1])
            latitude.append(x[2])
            longitude.append(x[3])
            text_review.append(x[4])
            quality.append(x[5])

        return place_id, name, latitude, longitude, text_review, quality
    except Exception as e:
        raise HTTPException(
            status_code=501, detail=f"Table Not Found : {e}")


def to_list_of_dict(data):
    list_of_dict = []
    for index, value in enumerate(data):
        list_of_dict.append({index: value})
    return list_of_dict


def to_update_prediction_results():
    # Here lies the codes
    return {'message': "Prediction Data Updated Successfully"}


def recommender_place(user_location: tuple):

    # test time
    st = time.time()

    # grab mysql connection database
    connection = connect_database()

    # fetch all necessary items
    place_id, name, latitude, longitude, text_review, quality = get_data_mysql(
        connection)

    # wrap items into dictionary
    df_temp = {
        'place_id': place_id,
        'name': name,
        'latitude': latitude,
        'longitude': longitude,
        'quality': quality
    }

    # convert dictionary to pandas dataframe
    df = pd.DataFrame(df_temp)

    # df = pd.read_csv(DATABASE_PATH_LOCATION)
    # df = df[['place_id', 'name', 'latitude', 'longitude', 'quality']]

    # membuat kolom coor yang berisi nilai latitude dan longitude
    df = df.assign(coor=list(zip(df.latitude, df.longitude)))

    # melakukan pengulangan untuk mendapatkan nilai jarak
    distances_km = []
    for row in df.itertuples(index=False):
        distances_km.append(distance_from(user_location, row[5]))

    # memasukkan nilai jarak ke kolom distance_from_user
    df = df.assign(distance_from_user=distances_km)

    df_distance_selected = df.sort_values(
        ['distance_from_user'], ascending=[True]).head(MAX_ITEM+ADD_ITEM)

    # menginisiasi model
    # model = tf.saved_model.load(MODEL_PATH)

    # # melakukan prediksi sentimen
    # sentiment = []
    # for sentence in df_distance_selected['text_review']:
    #     if len(str(sentence)) <= 1:
    #         sentiment.append(0.0)
    #     else:
    #         prediction = model([[sentence]]).numpy()[0][0]
    #         sentiment.append(prediction)

    # # memasukkan hasil prediksi ke dalam kolom quality
    # df_distance_selected = df_distance_selected.assign(quality=sentiment)

    # mengurutkan data berdasarkan prediksi sentimen
    df_quality_selected = df_distance_selected.sort_values(
        ['quality'], ascending=[True]).head(MAX_ITEM)

    # mengembalikan nilai respons
    ids = to_list_of_dict(df_quality_selected.loc[:, "place_id"])
    names = to_list_of_dict(df_quality_selected.loc[:, "name"])

    # end test time
    et = time.time()
    elapsed_time = et - st

    return {"places_id": ids, "places_name": names, "timez": elapsed_time}

# notes for commit :
# changes in algoritm, now predicting outside recommender place
# add variables to adjust outputs
