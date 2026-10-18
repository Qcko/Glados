import ctypes
import subprocess
import sys
import threading
import time
import uuid
from ctypes import wintypes

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named pipes need Windows")

if sys.platform == "win32":
    from glados.mcp import pipe_shim

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

_INVALID_HANDLE = ctypes.c_void_p(-1).value
_OTHER_SID = "S-1-5-21-1-2-3-500"


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p), ("bInheritHandle", wintypes.BOOL)]


@pytest.fixture(scope="module", autouse=True)
def _declare_win32():
    handle, dword, ptr = wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p
    _kernel32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, dword, dword, dword, dword, dword, dword, ptr]
    _kernel32.CreateNamedPipeW.restype = handle
    _kernel32.ConnectNamedPipe.argtypes = [handle, ptr]
    _kernel32.DisconnectNamedPipe.argtypes = [handle]
    _kernel32.ReadFile.argtypes = [handle, ptr, dword, ctypes.POINTER(dword), ptr]
    _kernel32.WriteFile.argtypes = [handle, ctypes.c_char_p, dword, ctypes.POINTER(dword), ptr]
    _kernel32.CloseHandle.argtypes = [handle]
    _kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, dword, dword, ptr, dword, dword, handle]
    _kernel32.CreateFileW.restype = handle
    _kernel32.GetCurrentProcess.restype = handle
    _advapi32.OpenProcessToken.argtypes = [handle, dword, ctypes.POINTER(handle)]
    _advapi32.GetTokenInformation.argtypes = [handle, ctypes.c_int, ptr, dword, ctypes.POINTER(dword)]
    _advapi32.ConvertSidToStringSidW.argtypes = [ptr, ctypes.POINTER(wintypes.LPWSTR)]
    _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, dword, ctypes.POINTER(ptr), ptr,
    ]


def _current_user_sid() -> str:
    token = wintypes.HANDLE()
    assert _advapi32.OpenProcessToken(_kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token))
    buffer = ctypes.create_string_buffer(256)
    size = wintypes.DWORD()
    assert _advapi32.GetTokenInformation(token, 1, buffer, len(buffer), ctypes.byref(size))
    _kernel32.CloseHandle(token)
    sid = ctypes.c_void_p.from_buffer(buffer).value
    text = wintypes.LPWSTR()
    assert _advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text))
    return text.value


class _EchoServer:
    def __init__(
        self, max_instances: int = 1, sddl: str | None = None, sessions: int = 1, hang_up_on_first_read: bool = False,
    ) -> None:
        self._hang_up_on_first_read = hang_up_on_first_read
        self.name = f"glados-test-{uuid.uuid4().hex}"
        self.received = bytearray()
        self.connected = threading.Event()
        self._handle = _create_server_pipe(self.name, max_instances, sddl)
        self._sessions = sessions
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        for _ in range(self._sessions):
            _kernel32.ConnectNamedPipe(self._handle, None)
            self.connected.set()
            self._echo_until_closed()
            _kernel32.DisconnectNamedPipe(self._handle)
        _kernel32.CloseHandle(self._handle)

    def _echo_until_closed(self) -> None:
        buffer = ctypes.create_string_buffer(4096)
        count = wintypes.DWORD()
        while _kernel32.ReadFile(self._handle, buffer, len(buffer), ctypes.byref(count), None) and count.value:
            data = buffer.raw[:count.value]
            self.received += data
            if self._hang_up_on_first_read:
                return
            written = wintypes.DWORD()
            _kernel32.WriteFile(self._handle, data, len(data), ctypes.byref(written), None)

    def join(self) -> None:
        self._thread.join(timeout=10)


def _create_server_pipe(name: str, max_instances: int, sddl: str | None) -> int:
    attributes = None
    if sddl:
        descriptor = ctypes.c_void_p()
        assert _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None)
        attributes = ctypes.byref(_SecurityAttributes(ctypes.sizeof(_SecurityAttributes), descriptor, False))
    handle = _kernel32.CreateNamedPipeW(
        "\\\\.\\pipe\\" + name, 0x3 | 0x00080000, 0x8, max_instances, 4096, 4096, 0, attributes,
    )
    assert handle != _INVALID_HANDLE, ctypes.FormatError(ctypes.get_last_error())
    return handle


def _run_shim(pipe: str, server_sid: str, stdin: bytes, wait_s: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "glados.mcp.pipe_shim", "--pipe", pipe, "--server-sid", server_sid, "--wait-s", str(wait_s)],
        input=stdin, capture_output=True, timeout=30,
    )


def _feed(shim: subprocess.Popen, payload: bytes) -> None:
    shim.stdin.write(payload)
    shim.stdin.flush()


def _echo_through_shim(pipe: str, server_sid: str, payload: bytes, wait_s: float = 5.0) -> tuple[int, bytes]:
    shim = subprocess.Popen(
        [sys.executable, "-m", "glados.mcp.pipe_shim", "--pipe", pipe, "--server-sid", server_sid, "--wait-s", str(wait_s)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        writer = threading.Thread(target=_feed, args=(shim, payload), daemon=True)
        writer.start()
        echoed = b""
        while len(echoed) < len(payload) and (chunk := shim.stdout.read1(65536)):
            echoed += chunk
        writer.join(timeout=10)
        shim.stdin.close()
        assert shim.wait(timeout=10) == pipe_shim.EXIT_OK, shim.stderr.read()
        return shim.returncode, echoed
    finally:
        shim.kill()


def test_relays_bytes_both_ways_and_exits_on_stdin_eof():
    server = _EchoServer()
    payload = b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n' * 2000
    _, echoed = _echo_through_shim(server.name, _current_user_sid(), payload)
    server.join()
    assert echoed == payload
    assert bytes(server.received) == payload


def test_exits_when_the_server_ends_the_session():
    server = _EchoServer(hang_up_on_first_read=True)
    shim = subprocess.Popen(
        [sys.executable, "-m", "glados.mcp.pipe_shim", "--pipe", server.name, "--server-sid", _current_user_sid()],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _feed(shim, b"x")
    try:
        assert shim.wait(timeout=10) == pipe_shim.EXIT_OK
    finally:
        shim.kill()
        shim.stdin.close()


def test_opens_with_read_and_write_data_only():
    sid = _current_user_sid()
    server = _EchoServer(sddl=f"D:P(A;;0x12008B;;;{sid})")
    _, echoed = _echo_through_shim(server.name, sid, b"hello\n")
    server.join()
    assert echoed == b"hello\n"


def test_refuses_a_pipe_owned_by_another_account_before_sending_a_byte():
    server = _EchoServer()
    result = _run_shim(server.name, _OTHER_SID, b"secret\n")
    server.join()
    assert result.returncode == pipe_shim.EXIT_WRONG_SERVER
    assert b"refusing" in result.stderr
    assert server.received == b""


def test_missing_pipe_fails_fast():
    started = time.monotonic()
    result = _run_shim(f"glados-test-absent-{uuid.uuid4().hex}", _current_user_sid(), b"", wait_s=20)
    assert result.returncode == pipe_shim.EXIT_UNREACHABLE
    assert time.monotonic() - started < 10


def test_waits_for_a_busy_pipe_to_free_up():
    server = _EchoServer(sessions=2)
    first = _kernel32.CreateFileW("\\\\.\\pipe\\" + server.name, 0x12008B, 0, None, 3, 0, None)
    assert first != _INVALID_HANDLE
    threading.Timer(1.0, _kernel32.CloseHandle, args=(first,)).start()
    _, echoed = _echo_through_shim(server.name, _current_user_sid(), b"after busy\n", wait_s=10)
    server.join()
    assert echoed == b"after busy\n"


def test_busy_pipe_gives_up_at_the_deadline():
    server = _EchoServer()
    first = _kernel32.CreateFileW("\\\\.\\pipe\\" + server.name, 0x12008B, 0, None, 3, 0, None)
    assert first != _INVALID_HANDLE
    try:
        result = _run_shim(server.name, _current_user_sid(), b"", wait_s=1)
    finally:
        _kernel32.CloseHandle(first)
    server.join()
    assert result.returncode == pipe_shim.EXIT_UNREACHABLE
    assert b"busy" in result.stderr


@pytest.mark.parametrize("args", [["--server-sid", "S-1-5-18"], ["--server-sid", _OTHER_SID, "--pipe", "bad\\name"]])
def test_rejects_bad_arguments(args):
    assert pipe_shim.main(args) == pipe_shim.EXIT_USAGE
