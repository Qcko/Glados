"""TLS arg resolution for the server entrypoint (`glados.main`).

`_tls_args` turns the [server] tls_certfile/tls_keyfile pair into uvicorn ssl
kwargs. The half-configured case must fail loudly rather than silently serve
cleartext on a LAN bind."""

from __future__ import annotations

import pytest

from glados import _tls_args


def test_no_tls_returns_empty() -> None:
    assert _tls_args("", "") == {}


def test_both_set_returns_ssl_kwargs() -> None:
    assert _tls_args("cert.pem", "key.pem") == {
        "ssl_certfile": "cert.pem",
        "ssl_keyfile": "key.pem",
    }


def test_cert_without_key_is_an_error() -> None:
    with pytest.raises(SystemExit, match="tls_keyfile"):
        _tls_args("cert.pem", "")


def test_key_without_cert_is_an_error() -> None:
    with pytest.raises(SystemExit, match="tls_certfile"):
        _tls_args("", "key.pem")
