def main() -> None:
    import uvicorn

    uvicorn.run("glados.core.server:app", host="127.0.0.1", port=8765, reload=False)
