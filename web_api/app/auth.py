# pylint: disable=all # type: ignore
import os

from fastapi import Request, Response
from google.auth.transport import requests

# from google.cloud import iap_v1
# from google.iam.v1 import iam_policy_pb2
from google.oauth2 import id_token


async def check_authorized_user(request: Request, call_next):
    if request.method != "OPTIONS" and request.url.path != "/healthz":
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
