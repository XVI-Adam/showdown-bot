import pytest

from psbot.config import BATTLE_FORMAT, local_server_configuration


def test_battle_format_is_gen9_randoms() -> None:
    assert BATTLE_FORMAT == "gen9randombattle"


def test_local_server_configuration_builds_loopback_websocket_url() -> None:
    config = local_server_configuration(port=8123)
    assert config.websocket_url == "ws://127.0.0.1:8123/showdown/websocket"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "sim3.psim.us"])
def test_local_server_configuration_rejects_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        local_server_configuration(host=host)


@pytest.mark.parametrize("port", [0, -1, 70000])
def test_local_server_configuration_rejects_bad_ports(port: int) -> None:
    with pytest.raises(ValueError, match="invalid port"):
        local_server_configuration(port=port)
