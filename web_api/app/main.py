# type: ignore
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .annotations import api as annotations_api
from .collections import api as collections_api
from .layer_groups import api as layer_groups_api
from .layers import api as layers_api

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/annotations", annotations_api)
app.mount("/collections", collections_api)
app.mount("/layer_groups", layer_groups_api)
app.mount("/layers", layers_api)


@app.get("/")
async def index():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    message = f"Hello world! From FastAPI running on Uvicorn. Using Python {version}"
    return {"message": message}
