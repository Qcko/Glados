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
    uvicorn.run("glados.core.server:app", host=server.host, port=server.port, reload=False)
