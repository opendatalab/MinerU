# Copyright (c) Opendatalab. All rights reserved.
from fastapi import FastAPI, HTTPException
from loguru import logger


PUBLIC_HTTP_CLIENT_DISABLED_DETAIL = (
    "Publicly exposed API disables *-http-client backends and server_url by "
    "default. Rebind to 127.0.0.1 or start with "
    "--allow-public-http-client if you understand the SSRF risk."
)


def is_public_bind_host(host: str) -> bool:
    return host in {"0.0.0.0", "::"}


def configure_public_http_client_policy(
    app: FastAPI,
    *,
    public_bind_exposed: bool,
    allow_public_http_client: bool,
) -> None:
    app.state.public_bind_exposed = public_bind_exposed
    app.state.allow_public_http_client = allow_public_http_client


def validate_public_http_client_request(
    *,
    public_bind_exposed: bool,
    allow_public_http_client: bool,
    backend: str,
    server_url: str | None,
) -> None:
    if not public_bind_exposed or allow_public_http_client:
        return
    if backend.endswith("-http-client") or bool(server_url and server_url.strip()):
        raise HTTPException(status_code=400, detail=PUBLIC_HTTP_CLIENT_DISABLED_DETAIL)


def warn_if_public_http_client_policy(
    *,
    service_name: str,
    host: str,
    allow_public_http_client: bool,
) -> None:
    if not is_public_bind_host(host):
        return
    if allow_public_http_client:
        logger.warning(
            "MinerU {} is listening on {} with --allow-public-http-client enabled. "
            "Requests may supply remote HTTP inference endpoints and turn the service "
            "into an externally driven outbound request primitive, creating SSRF and "
            "internal network probing risk.",
            service_name,
            host,
        )
        return
    logger.warning(
        "MinerU {} is listening on {}. Disabling *-http-client backends and "
        "server_url by default because these inputs let callers choose remote HTTP "
        "inference endpoints; when the API is publicly reachable, that creates SSRF "
        "and internal network probing risk.",
        service_name,
        host,
    )
