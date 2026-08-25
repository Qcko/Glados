"""The file-only marker keeps user content off stderr.

The claim-vocabulary log line carries a spoken reply verbatim -- product names,
quantities. The rotating file already sits beside `traces/`, which holds far
more; stderr is uvicorn's console and can be captured anywhere a service
manager decides, so it is the surface worth keeping clean.
"""

from __future__ import annotations

import logging

from glados.core.logging_setup import FILE_ONLY, _KeepFileOnlyOffStderr


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="glados.core.organizer", level=logging.INFO, pathname=__file__,
        lineno=1, msg="claim-vocab: reply=%r", args=("Took the milk off.",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_a_marked_record_is_kept_off_stderr() -> None:
    assert _KeepFileOnlyOffStderr().filter(_record(**{FILE_ONLY: True})) is False


def test_an_unmarked_record_still_reaches_stderr() -> None:
    # The filter must not quietly silence ordinary operational logging.
    assert _KeepFileOnlyOffStderr().filter(_record()) is True


def test_the_filter_is_installed_on_stderr_only(tmp_path, monkeypatch) -> None:
    """Guards the wiring, not just the predicate: a filter attached to the file
    handler by mistake would drop the line from the one place it belongs."""
    import sys

    import glados.core.logging_setup as ls

    monkeypatch.setenv("GLADOS_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(ls, "_configured", False)
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    try:
        ls.setup_logging()
        by_kind = {type(h).__name__: h for h in logging.getLogger().handlers}
        stream = by_kind["StreamHandler"]
        rotating = by_kind["RotatingFileHandler"]
        assert any(isinstance(f, ls._KeepFileOnlyOffStderr) for f in stream.filters)
        assert not any(isinstance(f, ls._KeepFileOnlyOffStderr) for f in rotating.filters)
    finally:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
        for h in saved:
            root.addHandler(h)
        root.setLevel(saved_level)
        ls._configured = False
