"""Fail closed if a heavy workflow regains an automatic entry point. AP_FLAKE8_CLEAN"""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())


class WorkflowPolicyTests(unittest.TestCase):
    def test_heavy_workflows_only_manual_or_reusable(self):
        for name in CONFIG['heavy_workflows']:
            with self.subTest(workflow=name):
                text = (ROOT / 'workflows' / name).read_text()
                block = re.search(r"^'?on'?\s*:[\s\S]*?(?=^[A-Za-z][A-Za-z_-]*:)", text, re.M).group()
                self.assertIn('workflow_dispatch:', block)
                for trigger in ('push:', 'pull_request:', 'schedule:', 'workflow_run:', 'release:', 'repository_dispatch:'):
                    self.assertNotIn(trigger, block)

    def test_automatic_recipes_reconstruct_pinned_source(self):
        for name in CONFIG['automatic_workflows']:
            text = (ROOT / 'workflows' / name).read_text()
            self.assertIn('workflow_call:', text)
            self.assertIn('RELEASE_MANIFEST: ${{ inputs.release_manifest }}', text)
            self.assertIn('python3 .github/release_policy/policy.py materialize', text)
            self.assertIn('ref: ${{ inputs.source_ref || github.sha }}', text)

    def test_build_is_conditional_and_manual_sync_defaults_to_sync_only(self):
        text = (ROOT / 'workflows/sync-upstream.yml').read_text()
        self.assertIn("if: needs.sync.outputs.build == 'true'", text)
        self.assertIn('default: sync', text)
        self.assertIn("github.event_name == 'schedule' || inputs.action == 'check-releases'", text)
        self.assertNotIn('workflow_run:', text)
        self.assertIn('cancel-in-progress: false', text)

    def test_publication_interface_does_not_rebuild_or_publish(self):
        text = (ROOT / 'workflows/promote-release.yml').read_text()
        self.assertIn('evidence.py verify', text)
        self.assertNotIn('materialize', text)
        self.assertNotIn('workflow_call:', text)
        self.assertIn('exit 1', text)

    def test_murmur_fixture_firmware_is_not_uploaded(self):
        path = ROOT / 'workflows/murmur.yml'
        if path.exists():
            text = path.read_text()
            self.assertIn('ci-only-not-a-secret', text)
            self.assertNotIn('upload-artifact', text)
            self.assertNotIn('gh release', text)

    def test_inherited_owner_guards_remain(self):
        for filename in ('release-exe.yml', 'release-pypi.yml'):
            path = ROOT / 'workflows' / filename
            if path.exists():
                self.assertIn("github.repository == 'jnz/INSLIB'", path.read_text())


if __name__ == '__main__':
    unittest.main()
