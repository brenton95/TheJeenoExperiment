from __future__ import annotations

from unittest.mock import patch

from jeenom.operator_station import run_repl


class InterruptingSession:
    def __init__(self) -> None:
        self.closed = False

    def startup(self) -> str:
        return "READY"

    def handle_utterance(self, utterance: str) -> str:
        raise KeyboardInterrupt

    def close(self) -> None:
        self.closed = True


def test_ctrl_c_during_command_exits_cleanly_and_closes_session(capsys):
    session = InterruptingSession()

    with patch("builtins.input", return_value="go to the blue door"):
        exit_code = run_repl(session)

    output = capsys.readouterr().out
    assert exit_code == 130
    assert "INTERRUPTED" in output
    assert session.closed
