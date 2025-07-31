import perceval.providers.scaleway as scw
import perceval as pcvl
import os

from perceval.providers.scaleway.scaleway_session import _ENDPOINT_URL


def build_session(name: str, platform: str) -> pcvl.ISession:
    session = scw.Session(
        deduplication_id=name,
        platform=platform,
        project_id=os.environ["SCW_PROJECT_ID"],
        token=os.environ["SCW_SECRET_KEY"],
        url=os.environ.get("SCW_API_GATEWAY", _ENDPOINT_URL),
    )

    return session
