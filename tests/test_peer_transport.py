from __future__ import annotations

import os
import socket
import threading
import unittest

from dllm.model import _decode_tensor, _encode_tensor
from dllm.peer import _read_envelope, _write_envelope, cleanup_tensor_payloads


class PeerTransportTests(unittest.TestCase):
    def test_json_tensor_envelope_inlines_raw_buffer_for_legacy_peer(self) -> None:
        import torch

        tensor = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        message = {
            "command": "forward_shard",
            "payload": {"hidden_state": _encode_tensor(tensor, raw=True)},
        }
        left, right = socket.socketpair()
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                _write_envelope(left, message, binary_tensors=False)
            except BaseException as exc:  # pragma: no cover - surfaced after join
                errors.append(exc)
            finally:
                left.close()

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            received = _read_envelope(right, spool_attachments=False)
            thread.join()
            if errors:
                raise errors[0]

            hidden_state = received["payload"]["hidden_state"]
            decoded = _decode_tensor(hidden_state, torch, "cpu")

            self.assertIsInstance(hidden_state["buffer"], str)
            self.assertTrue(torch.equal(decoded, tensor))
        finally:
            right.close()

    def test_binary_tensor_envelope_spools_and_round_trips(self) -> None:
        import torch

        tensor = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
        message = {
            "command": "forward_shard",
            "binary_tensors": True,
            "payload": {"hidden_state": _encode_tensor(tensor, raw=True)},
        }
        left, right = socket.socketpair()
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                _write_envelope(left, message, binary_tensors=True)
            except BaseException as exc:  # pragma: no cover - surfaced after join
                errors.append(exc)
            finally:
                left.close()

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            received = _read_envelope(right, spool_attachments=True)
            thread.join()
            if errors:
                raise errors[0]

            hidden_state = received["payload"]["hidden_state"]
            buffer_path = hidden_state["buffer_path"]
            decoded = _decode_tensor(hidden_state, torch, "cpu")

            self.assertTrue(os.path.exists(buffer_path))
            self.assertTrue(torch.equal(decoded, tensor))
            cleanup_tensor_payloads(received)
            self.assertFalse(os.path.exists(buffer_path))
        finally:
            right.close()


if __name__ == "__main__":
    unittest.main()
