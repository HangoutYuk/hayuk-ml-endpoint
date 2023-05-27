from fastapi import FastAPI
from modules.recommender import *
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


@app.get("/test-lokasi/{loc}")
def test_lokasi(loc: str):
    return {"lokasi": pengubah_lat_long(loc)}


@app.get("/pwd")
def get_pwd():
    return {"now on": os.getcwd(),
            "don ask": os.path.dirname(os.path.realpath(__file__))}

uvicorn.run(app, host='0.0.0.0',port=8000)
