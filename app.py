from fastapi import FastAPI
from modules.recommender import *
from modules.recommender_speed import *
from metadata import *
import uvicorn

app = FastAPI(
    title=title_api,
    description=description_api,
    version=version_api,
    openapi_tags=tags_metadata_api
)


@app.get("/")
def test():
    return {"response": "Hello World"}


@app.get("/recommend/{loc_user}", tags=['Get Recommended Locations'])
async def recommendation(loc_user):
    user_loc = pengubah_lat_long(loc_user)
    ids_recommended_place = recommender_place(user_loc)
    return ids_recommended_place


@app.get("/recommend_speed/{loc_user}", tags=['Get Recommended Locations with Speed'])
async def recommendation(loc_user):
    user_loc = pengubah_lat_long(loc_user)
    ids_recommended_place = recommender_place_speed(user_loc)
    return ids_recommended_place


@app.get("/refresh_sentiment_data_speed", tags=['Get Refresh Sentiment with Speed'])
async def refresh():
    refresh_status = refresh_sentiment_data()
    return refresh_status

uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))
