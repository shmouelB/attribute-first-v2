"""Install the suite-wide network guard before application tests are imported."""

import socket
import unittest

try:
    from . import offline_guard
except ImportError:
    import offline_guard


offline_guard.install()


class OfflineGuardTests(unittest.TestCase):
    def test_ipv4_connections_are_rejected_before_the_operating_system_call(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaisesRegex(RuntimeError, "network access blocked"):
                sock.connect(("192.0.2.1", 443))
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
