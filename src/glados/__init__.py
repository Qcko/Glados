def main() -> None:
    import os
    from pathlib import Path

    import uvicorn

    from glados.core.config import ServerConfig, load_glados_config

    config_dir = Path(os.environ.get("GLADOS_CONFIG_DIR", "configs"))
    glados_toml = config_dir / "glados.toml"
    server = (
        load_glados_config(glados_toml).server
        if glados_toml.is_file()
        else ServerConfig()
    )
    ssl_args = _tls_args(server.tls_certfile, server.tls_keyfile)
    uvicorn.run(
        "glados.core.server:app",
        host=server.host,
        port=server.port,
        reload=False,
        **ssl_args,
    )


def _tls_args(certfile: str, keyfile: str) -> dict:
    """uvicorn ssl kwargs when TLS is configured, else empty (plain HTTP).

    Both must be set together: a half-configured pair is an operator mistake
    that would otherwise silently fall back to cleartext on a LAN bind."""
    if not certfile and not keyfile:
        return {}
    if not (certfile and keyfile):
        missing = "tls_keyfile" if certfile else "tls_certfile"
        raise SystemExit(
            f"TLS misconfigured: {missing} is empty but the other is set; "
            f"set both (GLADOS_TLS_CERT + GLADOS_TLS_KEY, or [server] tls_* in "
            f"glados.toml) or neither."
        )
    return {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
