from __future__ import annotations

import unittest

from dllm.config import PeerSpec, parse_peers
from dllm.discovery import merge_peers


class DiscoveryTests(unittest.TestCase):
    def test_merge_peers_keeps_manual_and_adds_discovered(self) -> None:
        manual = tuple(parse_peers("node-b@127.0.0.1:8765"))
        discovered = (
            PeerSpec(name="node-b-copy", host="127.0.0.1", port=8765),
            PeerSpec(name="node-c", host="127.0.0.2", port=8765),
        )
        merged = merge_peers(manual, discovered, local_node_name="node-a")
        self.assertEqual([peer.name for peer in merged], ["node-b", "node-c"])


if __name__ == "__main__":
    unittest.main()
