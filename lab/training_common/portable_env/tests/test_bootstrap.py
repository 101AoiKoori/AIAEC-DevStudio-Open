"""Contract tests for incompatible interpreters and incomplete copied dependencies."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portable_bootstrap', SOURCE / 'bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class CompatibilityTests(unittest.TestCase):
    def test_offline_removes_proxies_and_preserves_localhost(self):
        proxies = {name: 'http://127.0.0.1:7890' for name in
                   ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')}
        with patch.dict(os.environ, proxies), \
                patch.object(bootstrap.socket.socket, 'connect', return_value=None) as connect, \
                patch.object(bootstrap.socket.socket, 'connect_ex', return_value=0) as connect_ex:
            bootstrap.deny_network()
            for name in proxies:
                self.assertNotIn(name, os.environ)
            self.assertEqual(os.environ['YOLO_OFFLINE'], 'true')
            self.assertEqual(os.environ['NO_PROXY'], '*')
            self.assertEqual(os.environ['no_proxy'], '*')
            bootstrap.socket.socket.connect(object(), ('127.0.0.1', 6006))
            connect.assert_called_once()
            self.assertEqual(bootstrap.socket.socket.connect_ex(object(), ('::1', 6006)), 0)
            connect_ex.assert_called_once()
            with self.assertRaisesRegex(OSError, 'Offline mode'):
                bootstrap.socket.socket.connect(object(), ('example.com', 443))

    def test_minor_and_abi_mismatch_are_errors(self):
        actual = bootstrap.interpreter_info()
        for field, wrong in [('python_version', [3, 11]), ('bits', 32), ('extension_suffix', '.cp311-win_amd64.pyd')]:
            with self.subTest(field=field):
                manifest = dict(actual, **{field: wrong})
                with self.assertRaisesRegex(RuntimeError, 'Incompatible Python'):
                    bootstrap.validate_interpreter(manifest)

    def test_standard_compatible_interpreter_is_accepted(self):
        info = bootstrap.interpreter_info()
        self.assertEqual(bootstrap.validate_interpreter(info), info)

    def test_missing_or_truncated_package_file_is_error(self):
        parent = SOURCE / 'runs' / 'tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            vendor = root / 'site-packages'
            metadata = vendor / 'sample-1.0.dist-info'
            metadata.mkdir(parents=True)
            (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: sample\nVersion: 1.0\n', encoding='utf-8')
            (metadata / 'RECORD').write_text('sample.py,,4\n', encoding='utf-8')
            manifest = {'packages': {'sample': '1.0'}}
            with patch.object(bootstrap, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'missing or truncated'):
                    bootstrap.inventory(vendor, manifest, full=True)
                (vendor / 'sample.py').write_text('x', encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, 'missing or truncated'):
                    bootstrap.inventory(vendor, manifest, full=True)
                (vendor / 'sample.py').write_text('pass', encoding='utf-8')
                self.assertEqual(bootstrap.inventory(vendor, manifest, full=True)['record_files_checked'], 1)
                with self.assertRaisesRegex(RuntimeError, 'versions differ'):
                    bootstrap.inventory(vendor, {'packages': {'sample': '2.0'}})


if __name__ == '__main__':
    unittest.main()
