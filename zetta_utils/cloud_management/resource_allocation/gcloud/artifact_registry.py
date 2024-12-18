from google.auth import default
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from zetta_utils import log

logger = log.get_logger("zetta_utils")


def check_image_exists(image: str) -> bool:
    image_path, tag = image.split(":")
    try:
        parts = image_path.split("/")
        hostname_parts = parts[0].split(".")
        location = hostname_parts[0].replace("-docker", "")
        project_id = parts[1]
        repository = parts[2]
        image_name = parts[3]
    except IndexError:
        logger.exception(
            "Expected format: '<location>-docker.pkg.dev/<project-id>/<repository>/<image-name>'"
        )
        return False

    credentials, _ = default()
    credentials.refresh(Request())
    service = build("artifactregistry", "v1", credentials=credentials)
    tag_name = (
        f"projects/{project_id}/locations/{location}/"
        f"repositories/{repository}/packages/{image_name}/tags/{tag}"
    )
    try:
        response = (
            service.projects()
            .locations()
            .repositories()
            .packages()
            .tags()
            .get(name=tag_name)
            .execute()
        )
        if response:
            return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "404" in str(e):
            logger.error(f"Tag '{tag}' does not exist for image '{image_name}'.")
        else:
            logger.error(f"Error checking image tag: {e}")
    return False
