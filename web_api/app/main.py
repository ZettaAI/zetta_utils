# pylint: disable=all # type: ignore
import os
import sys

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport import requests

# from google.cloud import iap_v1
# from google.iam.v1 import iam_policy_pb2
from google.oauth2 import id_token

from .annotations import api as annotations_api
from .collections import api as collections_api
from .layer_groups import api as layer_groups_api
from .layers import api as layers_api
from .painting import api as painting_api

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
app.mount("/painting", painting_api)


@app.middleware("http")
async def check_authorized_user(request: Request, call_next):
    if request.method != "OPTIONS":
        try:
            token = request.headers["authorization"].split()[-1]
        except (KeyError, IndexError):
            return Response(content="Missing auth token.", status_code=401)

        client_id = os.environ["OAUTH_CLIENT_ID"]
        try:
            idinfo = id_token.verify_oauth2_token(token, requests.Request(), client_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return Response(content=str(exc), status_code=401)

        if not idinfo["email"].endswith("@zetta.ai"):
            return Response(content="User not authorized.", status_code=401)
        #  user = f"user:{idinfo['email']}"
        # client = iap_v1.IdentityAwareProxyAdminServiceClient()

        # iap_resource = os.environ["IAP_RESOURCE"]
        # request = iam_policy_pb2.GetIamPolicyRequest(resource=iap_resource)
        # policy = client.get_iam_policy(request=request)
        # members = set(policy.bindings[0].members)
        # if user not in members:
        #     return Response(content="User not authorized.", status_code=401)

    response = await call_next(request)
    return response


@app.get("/")
async def index():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    message = f"Hello world! From FastAPI running on Uvicorn. Using Python {version}"
    return {"message": message}
