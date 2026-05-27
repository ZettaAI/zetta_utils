# pylint: disable=all # type: ignore
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport import requests

# from google.cloud import iap_v1
# from google.iam.v1 import iam_policy_pb2
from google.oauth2 import id_token

from .alignment import api as alignment_api
from .annotations import api as annotations_api
from .boot_self_check import assert_no_serviceaccount_token
from .collections import api as collections_api
from .layer_groups import api as layer_groups_api
from .layers import api as layers_api
from .painting import api as painting_api
from .precomputed_annotations import api as precomputed_annotations_api
from .run_spec import api as run_spec_api
from .segmentation import api as segmentation_api
from .session import api as session_api
from .tasks import api as tasks_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert_no_serviceaccount_token()
    yield


app = FastAPI(lifespan=lifespan)

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


def verify_zetta_ai_id_token(authorization: str) -> dict:
    """Verify a Bearer Google ID token and enforce the ``@zetta.ai`` domain.

    Splits the Bearer token out of the ``authorization`` header value, verifies
    it against ``OAUTH_CLIENT_ID``, and asserts the decoded email ends in
    ``@zetta.ai``. Raises ``HTTPException(401)`` on every failure, otherwise
    returns the decoded token info dict.

    :param authorization: the raw ``Authorization`` header value.
    :return: the decoded ID-token info.
    """
    try:
        token = authorization.split()[-1]
    except (AttributeError, IndexError):
        raise HTTPException(status_code=401, detail="Missing auth token.")

    client_id = os.environ["OAUTH_CLIENT_ID"]
    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), client_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise HTTPException(status_code=401, detail=str(exc))

    if not idinfo["email"].endswith("@zetta.ai"):
        raise HTTPException(status_code=401, detail="User not authorized.")
    return idinfo


@app.middleware("http")
async def check_authorized_user(request: Request, call_next):
    if request.method != "OPTIONS" and request.url.path != "/healthz":
        if "authorization" not in request.headers:
            return Response(content="Missing auth token.", status_code=401)
        try:
            verify_zetta_ai_id_token(request.headers["authorization"])
        except HTTPException as exc:
            return Response(content=exc.detail, status_code=exc.status_code)
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


@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}
