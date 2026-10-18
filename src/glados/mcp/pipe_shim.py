"""Relay stdin/stdout to a local named pipe, so a pipe-hosted MCP server looks like stdio.

GLaDOS spawns this as an ordinary stdio server. It opens the pipe with read + write-data
only, refuses a pipe whose owner is not the expected server account, and then copies bytes
both ways until either side ends.

Windows only, stdlib only (ctypes).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
import threading
import time
from ctypes import wintypes

PIPE_CLIENT_ACCESS = 0x0012008B
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3
EXIT_WRONG_SERVER = 4
EXIT_RELAY_FAILED = 5

_OPEN_EXISTING = 3
_FILE_FLAG_OVERLAPPED = 0x40000000
_SECURITY_SQOS_PRESENT = 0x00100000
_SECURITY_IDENTIFICATION = 0x00010000
_OPEN_FLAGS = _FILE_FLAG_OVERLAPPED | _SECURITY_SQOS_PRESENT | _SECURITY_IDENTIFICATION
_SE_KERNEL_OBJECT = 6
_OWNER_SECURITY_INFORMATION = 0x1
_ERROR_FILE_NOT_FOUND = 2
_ERROR_BROKEN_PIPE = 109
_ERROR_PIPE_BUSY = 231
_ERROR_PIPE_NOT_CONNECTED = 233
_PIPE_ENDED = frozenset({_ERROR_BROKEN_PIPE, _ERROR_PIPE_NOT_CONNECTED})
_ERROR_IO_PENDING = 997
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_CHUNK = 65536

_PIPE_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_ACCOUNT_SID = re.compile(r"S-1-5-21(-\d+){4}\Z")


class ShimError(Exception):
    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def main(argv: list[str] | None = None) -> int:
    try:
        options = _parse_args(argv)
        handle = open_pipe(pipe_path(options.pipe), options.wait_s)
        verify_owner(handle, options.server_sid)
    except ShimError as error:
        print(f"pipe_shim: {error}", file=sys.stderr, flush=True)
        return error.exit_code
    failure = relay(handle, sys.stdin.fileno(), sys.stdout.fileno())
    if failure:
        print(f"pipe_shim: relay failed: {failure}", file=sys.stderr, flush=True)
        return EXIT_RELAY_FAILED
    return EXIT_OK


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m glados.mcp.pipe_shim")
    parser.add_argument("--pipe", default="calendar-mcp")
    parser.add_argument("--server-sid", required=True)
    parser.add_argument("--wait-s", type=float, default=30.0)
    options = parser.parse_args(argv)
    if sys.platform != "win32":
        raise ShimError(EXIT_USAGE, "named pipes need Windows")
    if not _ACCOUNT_SID.match(options.server_sid):
        raise ShimError(EXIT_USAGE, "--server-sid is not an account SID")
    return options


def pipe_path(pipe_name: str) -> str:
    if not _PIPE_NAME.match(pipe_name):
        raise ShimError(EXIT_USAGE, "--pipe must be 1-64 of [A-Za-z0-9._-]")
    return "\\\\.\\pipe\\" + pipe_name


def open_pipe(path: str, wait_s: float) -> int:
    deadline = time.monotonic() + wait_s
    while True:
        handle = _kernel32.CreateFileW(path, PIPE_CLIENT_ACCESS, 0, None, _OPEN_EXISTING, _OPEN_FLAGS, None)
        if handle != _INVALID_HANDLE:
            return handle
        error = ctypes.get_last_error()
        if error != _ERROR_PIPE_BUSY:
            raise ShimError(EXIT_UNREACHABLE, _open_failure(path, error))
        _wait_for_free_instance(path, deadline)


def _open_failure(path: str, error: int) -> str:
    if error == _ERROR_FILE_NOT_FOUND:
        return f"{path} does not exist (is the server running?)"
    return f"cannot open {path}: {ctypes.FormatError(error)}"


def _wait_for_free_instance(path: str, deadline: float) -> None:
    remaining_ms = int((deadline - time.monotonic()) * 1000)
    if remaining_ms <= 0:
        raise ShimError(EXIT_UNREACHABLE, f"{path} stayed busy")
    _kernel32.WaitNamedPipeW(path, remaining_ms)


def verify_owner(handle: int, expected_sid: str) -> None:
    owner = pipe_owner_sid(handle)
    if owner.upper() != expected_sid.upper():
        raise ShimError(EXIT_WRONG_SERVER, f"pipe is owned by {owner}, not {expected_sid}; refusing it")


def pipe_owner_sid(handle: int) -> str:
    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    status = _advapi32.GetSecurityInfo(
        handle, _SE_KERNEL_OBJECT, _OWNER_SECURITY_INFORMATION,
        ctypes.byref(owner), None, None, None, ctypes.byref(descriptor),
    )
    if status != 0:
        raise ShimError(EXIT_WRONG_SERVER, f"cannot read the pipe owner: {ctypes.FormatError(status)}")
    try:
        return _sid_to_string(owner)
    finally:
        _kernel32.LocalFree(descriptor)


def _sid_to_string(sid: ctypes.c_void_p) -> str:
    text = wintypes.LPWSTR()
    if not _advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        raise ShimError(EXIT_WRONG_SERVER, "cannot format the pipe owner SID")
    try:
        return text.value
    finally:
        _kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))


def relay(handle: int, in_fd: int, out_fd: int) -> OSError | None:
    finished = threading.Event()
    failures: list[OSError] = []
    for pump in (_pump_input_to_pipe, _pump_pipe_to_output):
        threading.Thread(
            target=_run_until_done, args=(pump, handle, in_fd, out_fd, finished, failures), daemon=True,
        ).start()
    finished.wait()
    _kernel32.CancelIoEx(handle, None)
    return failures[0] if failures else None


def _run_until_done(pump, handle: int, in_fd: int, out_fd: int, finished: threading.Event, failures: list) -> None:
    try:
        pump(handle, in_fd, out_fd)
    except OSError as error:
        if not finished.is_set():
            failures.append(error)
    finally:
        finished.set()


def _pump_input_to_pipe(handle: int, in_fd: int, _out_fd: int) -> None:
    with _Overlapped() as overlapped:
        while chunk := os.read(in_fd, _CHUNK):
            _write_all(handle, chunk, overlapped)


def _pump_pipe_to_output(handle: int, _in_fd: int, out_fd: int) -> None:
    buffer = ctypes.create_string_buffer(_CHUNK)
    with _Overlapped() as overlapped:
        while count := _read_some(handle, buffer, overlapped):
            _write_fd_all(out_fd, buffer.raw[:count])


def _read_some(handle: int, buffer, overlapped: _Overlapped) -> int:
    started = _kernel32.ReadFile(handle, buffer, len(buffer), None, overlapped.fresh())
    return _finish_io(handle, started, overlapped)


def _write_all(handle: int, data: bytes, overlapped: _Overlapped) -> None:
    sent = 0
    while sent < len(data):
        remaining = data[sent:]
        started = _kernel32.WriteFile(handle, remaining, len(remaining), None, overlapped.fresh())
        count = _finish_io(handle, started, overlapped)
        if count == 0:
            raise OSError(_ERROR_BROKEN_PIPE, "pipe closed")
        sent += count


def _finish_io(handle: int, started: bool, overlapped: _Overlapped) -> int:
    if not started:
        error = ctypes.get_last_error()
        if error in _PIPE_ENDED:
            return 0
        if error != _ERROR_IO_PENDING:
            raise OSError(error, ctypes.FormatError(error))
    count = wintypes.DWORD()
    if not _kernel32.GetOverlappedResult(handle, overlapped.ref(), ctypes.byref(count), True):
        error = ctypes.get_last_error()
        if error in _PIPE_ENDED:
            return 0
        raise OSError(error, ctypes.FormatError(error))
    return count.value


def _write_fd_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


class _OverlappedStruct(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


class _Overlapped:
    def __enter__(self) -> _Overlapped:
        self._event = _kernel32.CreateEventW(None, True, False, None)
        if not self._event:
            raise OSError(ctypes.get_last_error(), "CreateEventW failed")
        self._struct = _OverlappedStruct()
        return self

    def __exit__(self, *_exc) -> None:
        _kernel32.CloseHandle(self._event)

    def fresh(self):
        self._struct = _OverlappedStruct(hEvent=self._event)
        return self.ref()

    def ref(self):
        return ctypes.byref(self._struct)


def _declare_win32() -> None:
    handle, dword, ptr = wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p
    _kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, dword, dword, ptr, dword, dword, handle]
    _kernel32.CreateFileW.restype = handle
    _kernel32.WaitNamedPipeW.argtypes = [wintypes.LPCWSTR, dword]
    _kernel32.WaitNamedPipeW.restype = wintypes.BOOL
    _kernel32.ReadFile.argtypes = [handle, ptr, dword, ptr, ptr]
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.WriteFile.argtypes = [handle, ctypes.c_char_p, dword, ptr, ptr]
    _kernel32.WriteFile.restype = wintypes.BOOL
    _kernel32.GetOverlappedResult.argtypes = [handle, ptr, ctypes.POINTER(dword), wintypes.BOOL]
    _kernel32.GetOverlappedResult.restype = wintypes.BOOL
    _kernel32.CreateEventW.argtypes = [ptr, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel32.CreateEventW.restype = handle
    _kernel32.CancelIoEx.argtypes = [handle, ptr]
    _kernel32.CloseHandle.argtypes = [handle]
    _kernel32.LocalFree.argtypes = [ptr]
    _kernel32.LocalFree.restype = ptr
    _advapi32.GetSecurityInfo.argtypes = [handle, ctypes.c_int, dword] + [ctypes.POINTER(ptr)] * 4 + [ctypes.POINTER(ptr)]
    _advapi32.GetSecurityInfo.restype = dword
    _advapi32.ConvertSidToStringSidW.argtypes = [ptr, ctypes.POINTER(wintypes.LPWSTR)]
    _advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL


if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _declare_win32()


if __name__ == "__main__":
    sys.exit(main())
