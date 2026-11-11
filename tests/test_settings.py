"""Settings defaults that affect who can reach the gateway."""

from config.settings import Settings


def test_gateway_listens_only_on_localhost_by_default(monkeypatch):
    # The admin endpoints have no authentication, so the default must not expose them to the network.
    monkeypatch.delenv("HOST", raising=False)
    assert Settings(_env_file=None).host == "127.0.0.1"


def test_host_can_still_be_opened_up_explicitly(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert Settings(_env_file=None).host == "0.0.0.0"
