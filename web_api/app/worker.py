# pylint: disable=all # type: ignore
from fastapi import FastAPI

from .auth import check_authorized_user
from .run_spec import api as run_spec_api

app = FastAPI()

app.middleware("http")(check_authorized_user)

app.mount("/run_spec", run_spec_api)


@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}
