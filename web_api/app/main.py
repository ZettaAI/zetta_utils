# pylint: disable=all # type: ignore
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .alignment import api as alignment_api
from .annotations import api as annotations_api
from .auth import check_authorized_user
from .collections import api as collections_api
from .layer_groups import api as layer_groups_api
from .layers import api as layers_api
from .painting import api as painting_api
from .precomputed_annotations import api as precomputed_annotations_api
from .run_spec import api as run_spec_api
from .segmentation import api as segmentation_api
from .session import api as session_api
from .tasks import api as tasks_api

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/alignment", alignment_api)
app.mount("/annotations", annotations_api)
app.mount("/collections", collections_api)
app.mount("/layer_groups", layer_groups_api)
app.mount("/layers", layers_api)
app.mount("/painting", painting_api)
app.mount("/precomputed", precomputed_annotations_api)
app.mount("/run_spec", run_spec_api)
app.mount("/segmentation", segmentation_api)
app.mount("/sessions", session_api)
app.mount("/tasks", tasks_api)


app.middleware("http")(check_authorized_user)


@app.get("/")
async def index():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    message = f"Hello world! From FastAPI running on Uvicorn. Using Python {version}"
    return {"message": message}


@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}
