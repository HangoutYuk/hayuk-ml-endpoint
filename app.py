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

uvicorn.run(app, host='0.0.0.0', port=8000)
