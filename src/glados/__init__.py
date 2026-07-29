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
    if not server.admin_port:
        uvicorn.run(
            "glados.core.server:app",
            host=server.host,
            port=server.port,
            reload=False,
            **ssl_args,
        )
        return
    _run_with_admin(server, ssl_args)


def _run_with_admin(server, ssl_args: dict) -> None:
    """Run the main app plus the loopback-only admin room-viewer in one event
    loop. The admin app is ALWAYS bound to 127.0.0.1 regardless of `host`, so
    the observe capability never reaches the LAN (ARCHITECTURE section 9). Both
    uvicorn servers share the process -- and thus the one Organizer + admin
    registries created when glados.core.server is imported."""
    import asyncio

    import uvicorn

    from glados.core.server import app, build_admin_app

    if server.admin_port == server.port:
        raise SystemExit(
            f"admin_port ({server.admin_port}) must differ from the main port "
            f"({server.port}); set GLADOS_ADMIN_PORT to a free port."
        )

    admin_app = build_admin_app(app)
    main_server = uvicorn.Server(
        uvicorn.Config(app, host=server.host, port=server.port, **ssl_args)
    )
    # Plain HTTP on loopback -- TLS adds nothing over a 127.0.0.1 bind, and the
    # admin secret is the defense-in-depth layer on top of loopback isolation.
    admin_server = uvicorn.Server(
        uvicorn.Config(admin_app, host="127.0.0.1", port=server.admin_port)
    )

    async def _serve_both() -> None:
        # Each Server.serve() installs its own signal handlers; the second
        # install wins, so a Ctrl-C only flips one server's should_exit. Wait
        # for whichever stops first (signal or crash), then tell the other to
        # exit too -- otherwise gather would hang forever on the survivor.
        tasks = [
            asyncio.create_task(main_server.serve()),
            asyncio.create_task(admin_server.serve()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        main_server.should_exit = True
        admin_server.should_exit = True
        await asyncio.gather(*tasks)

    asyncio.run(_serve_both())


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
