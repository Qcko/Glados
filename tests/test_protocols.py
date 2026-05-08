from pydantic import TypeAdapter

from glados.core.protocols import (
    AssistantDelta,
    ClientMessage,
    Hello,
    ServerMessage,
    UserText,
)


client_adapter = TypeAdapter(ClientMessage)
server_adapter = TypeAdapter(ServerMessage)


def test_client_hello_roundtrip() -> None:
    msg = Hello(client_id="desk-ui", room_id="desk", role="ui", token="t")
    parsed = client_adapter.validate_python(msg.model_dump())
    assert isinstance(parsed, Hello)
    assert parsed.role == "ui"


def test_client_user_text_roundtrip() -> None:
    parsed = client_adapter.validate_python({"type": "user_text", "text": "hi"})
    assert isinstance(parsed, UserText)
    assert parsed.text == "hi"


def test_server_assistant_delta_roundtrip() -> None:
    msg = AssistantDelta(session_id="s1", text="hello")
    parsed = server_adapter.validate_python(msg.model_dump())
    assert isinstance(parsed, AssistantDelta)
    assert parsed.session_id == "s1"


def test_unknown_type_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        client_adapter.validate_python({"type": "nope"})
