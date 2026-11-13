import socket

import pytest

from psbot.server import ServerUnavailableError, is_server_reachable, require_server


@pytest.fixture
def listening_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]


@pytest.fixture
def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return port  # socket is closed, so nothing is listening here


def test_reachable_when_something_listens(listening_port: int) -> None:
    assert is_server_reachable("127.0.0.1", listening_port)
    require_server("127.0.0.1", listening_port)  # does not raise


def test_unreachable_when_nothing_listens(closed_port: int) -> None:
    assert not is_server_reachable("127.0.0.1", closed_port, timeout=0.5)


def test_require_server_explains_how_to_start_it(closed_port: int) -> None:
    with pytest.raises(ServerUnavailableError, match="scripts/start_server.sh"):
        require_server("127.0.0.1", closed_port)
