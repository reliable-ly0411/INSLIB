"""Artifact evidence regression tests. AP_FLAKE8_CLEAN"""
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('evidence', Path(__file__).with_name('evidence.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class EvidenceTests(unittest.TestCase):
    def test_missing_digest_blocks(self):
        with self.assertRaises(ValueError):
            e.validate_digest({'id': 1, 'name': 'firmware'})

    def test_expired_artifact_blocks_instead_of_rebuilding(self):
        with self.assertRaisesRegex(ValueError, 'expired'):
            e.validate_digest({'id': 1, 'name': 'firmware', 'digest': 'sha256:' + 'a' * 64, 'expired': True})

    def test_upstream_stability_is_rechecked_before_promotion(self):
        manifest = {'upstream': 'test/repo', 'release_id': '1', 'tag': 'v1.0.0', 'upstream_sha': 'a' * 40}
        release = {'draft': False, 'prerelease': False, 'tag_name': 'v1.0.0'}
        refs = 'a' * 40 + ' refs/tags/v1.0.0\n'
        with patch.object(e, 'api', return_value=release), patch.object(e.subprocess, 'check_output', return_value=refs):
            e.verify_upstream(manifest)
        release['prerelease'] = True
        with patch.object(e, 'api', return_value=release), self.assertRaisesRegex(ValueError, 'no longer formal'):
            e.verify_upstream(manifest)

    def test_downloaded_bytes_are_verified(self):
        data = b'exact original artifact bytes'
        artifact = {'id': 1, 'name': 'firmware', 'digest': 'sha256:' + hashlib.sha256(data).hexdigest()}
        with patch.object(e.subprocess, 'check_output', return_value=data):
            self.assertEqual(e.archive('test/repo', artifact), data)
        with patch.object(e.subprocess, 'check_output', return_value=b'modified bytes'):
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                e.archive('test/repo', artifact)


if __name__ == '__main__':
    unittest.main()
