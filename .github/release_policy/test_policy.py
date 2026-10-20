"""Policy regression tests. AP_FLAKE8_CLEAN"""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('policy', Path(__file__).with_name('policy.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SHA = 'a' * 40


def release(rid=1, tag='v1.0.0', **kwargs):
    return dict(id=rid, tag_name=tag, draft=False, prerelease=False,
                published_at='2026-10-21T00:00:00Z', **kwargs)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = {'families': {'stable': r'^v\d+\.\d+\.\d+$'}, 'baseline': {},
                       'bootstrap_at': '2026-10-20T00:00:00Z', 'max_attempts': 3}
        self.state = p.new_state(self.config)

    def select(self, values, sha=SHA, retry_id=None):
        return p.select_release(values, self.state, self.config, lambda tag: sha, retry_id)

    def test_new_formal_release(self):
        selected = self.select([release()])
        self.assertEqual(selected[0], '1')
        self.assertEqual(selected[1]['upstream_sha'], SHA)

    def test_bootstrap_stable_is_not_backfilled(self):
        self.config['baseline'] = {'1': {'tag': 'v1.0.0', 'status': 'baseline', 'upstream_sha': SHA}}
        self.state = p.new_state(self.config)
        self.assertIsNone(self.select([release()]))

    def test_unknown_historical_release_is_baselined(self):
        r = release()
        r['published_at'] = '2026-01-01T00:00:00Z'
        self.assertIsNone(self.select([r]))
        self.assertEqual(self.state['releases']['1']['status'], 'baseline')

    def test_draft_and_prerelease_do_not_build(self):
        for key in ('draft', 'prerelease'):
            r = release()
            r[key] = True
            self.assertIsNone(self.select([r]))

    def test_prerelease_promotion_with_old_date(self):
        r = release()
        r.update(prerelease=True, published_at='2026-01-01T00:00:00Z')
        self.assertIsNone(self.select([r]))
        r['prerelease'] = False
        self.assertIsNotNone(self.select([r]))

    def test_unrecognized_tag_fails_closed(self):
        self.assertIsNone(self.select([release(tag='nightly')]))

    def test_retry_keeps_original_manifest(self):
        manifest = {'downstream_sha': 'b' * 40, 'source_sha': 'c' * 40}
        self.state['releases']['1'] = dict(
            tag='v1.0.0', status='failed', upstream_sha=SHA, attempts=1, manifest=manifest)
        selected = self.select([release()])
        self.assertEqual(selected[1]['manifest'], manifest)

    def test_retry_bound_and_explicit_retry(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='failed', upstream_sha=SHA, attempts=3)
        self.assertIsNone(self.select([release()]))
        self.assertIsNotNone(self.select([release()], retry_id='1'))

    def test_success_is_deduplicated(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='built', upstream_sha=SHA)
        self.assertIsNone(self.select([release()]))

    def test_moved_tag_fails_closed(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='failed', upstream_sha='b' * 40)
        self.assertIsNone(self.select([release()]))
        self.assertEqual(self.state['releases']['1']['status'], 'blocked')

    def test_renamed_release_fails_closed(self):
        self.state['releases']['1'] = dict(tag='v0.9.0', status='failed', upstream_sha=SHA)
        self.assertIsNone(self.select([release()]))
        self.assertEqual(self.state['releases']['1']['status'], 'blocked')

    def test_one_release_per_run_and_family_isolation(self):
        self.config['families'] = {'3.x': r'^3\.\d+\.\d+$', '4.x': r'^4\.\d+\.\d+$'}
        self.state['releases']['1'] = dict(tag='3.6.0', status='baseline', upstream_sha=SHA)
        selected = self.select([release(1, '3.6.0'), release(2, '4.1.0'), release(3, '3.7.0')])
        self.assertEqual(selected[0], '2')
        self.assertEqual(selected[1]['family'], '4.x')

    def test_recreated_release_id_cannot_reuse_tag(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='built', upstream_sha=SHA)
        self.assertIsNone(self.select([release(2)]))
        self.assertEqual(self.state['releases']['2']['status'], 'blocked')

    def test_historical_tag_movement_is_reported_blocked(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='baseline', upstream_sha='b' * 40)
        self.assertIsNone(self.select([release()]))
        self.assertEqual(self.state['releases']['1']['status'], 'blocked')

    def test_deleted_release_is_blocked(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='built', upstream_sha=SHA)
        self.assertIsNone(self.select([]))
        self.assertEqual(self.state['releases']['1']['status'], 'blocked')

    def test_revoked_stable_status_is_blocked(self):
        self.state['releases']['1'] = dict(tag='v1.0.0', status='built', upstream_sha=SHA)
        r = release()
        r['prerelease'] = True
        self.assertIsNone(self.select([r]))
        self.assertEqual(self.state['releases']['1']['status'], 'blocked')

    def test_invalid_tag_commit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.select([release()], sha='not-a-sha')

    def test_ambiguous_family_is_rejected(self):
        self.config['families']['duplicate'] = r'^v.*'
        with self.assertRaises(ValueError):
            self.select([release()])


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@localhost')
        (self.repo / 'code.txt').write_text('released\n')
        self.commit('release')
        self.release_sha = self.git('rev-parse', 'HEAD')
        self.git('tag', '-a', 'v1.0.0', '-m', 'annotated tag')
        (self.repo / 'unreleased.txt').write_text('must not leak into released source\n')
        self.commit('unreleased upstream change')
        self.base = self.git('rev-parse', 'HEAD')
        (self.repo / 'local.txt').write_text('downstream patch\n')
        for directory in p.CONTROL_PATHS:
            (self.repo / directory).mkdir(parents=True, exist_ok=True)
            (self.repo / directory / 'policy.txt').write_text('downstream policy\n')
        self.commit('downstream patch and policy')
        downstream = self.git('rev-parse', 'HEAD')
        self.manifest = dict(upstream='test/upstream', upstream_sha=self.release_sha,
                             downstream_sha=downstream, patch_base=self.base, tag='v1.0.0',
                             published_at='2026-10-21T00:00:00Z')
        self.manifest['patch_sha256'] = p.hashlib.sha256(p.patch_bytes(self.manifest, self.repo)).hexdigest()
        self.real_git = p.git

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def commit(self, message):
        self.git('add', '.')
        self.git('commit', '-m', message)

    def local_fetch(self, *args, **kwargs):
        if args[0] == 'fetch':
            return self.real_git('fetch', '--no-tags', str(self.repo), 'refs/tags/v1.0.0', **kwargs)
        return self.real_git(*args, **kwargs)

    def test_materialization_excludes_unreleased_head_and_is_reproducible(self):
        with patch.object(p, 'git', self.local_fetch):
            sha = p.materialize(self.manifest, self.repo, update_submodules=False)
            self.assertFalse((self.repo / 'unreleased.txt').exists())
            self.assertEqual((self.repo / 'local.txt').read_text(), 'downstream patch\n')
            self.assertTrue((self.repo / '.github/workflows/policy.txt').exists())
            self.manifest['source_sha'] = sha
            self.assertEqual(p.materialize(self.manifest, self.repo, update_submodules=False), sha)

    def test_patch_digest_tampering_is_rejected(self):
        self.manifest['patch_sha256'] = '0' * 64
        with patch.object(p, 'git', self.local_fetch), self.assertRaisesRegex(ValueError, 'digest'):
            p.materialize(self.manifest, self.repo, update_submodules=False)

    def test_moved_annotated_tag_is_rejected(self):
        self.git('tag', '-f', 'v1.0.0')
        with patch.object(p, 'git', self.local_fetch), self.assertRaisesRegex(ValueError, 'tag moved'):
            p.materialize(self.manifest, self.repo, update_submodules=False)

    def test_source_sha_tampering_is_rejected(self):
        self.manifest['source_sha'] = 'f' * 40
        with patch.object(p, 'git', self.local_fetch), self.assertRaisesRegex(ValueError, 'source commit'):
            p.materialize(self.manifest, self.repo, update_submodules=False)


class StateTests(unittest.TestCase):
    def test_state_branch_is_append_only_and_unchanged_state_does_not_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = str(Path(tmp) / 'remote.git')
            work = Path(tmp) / 'work'
            subprocess.run(['git', 'init', '--bare', remote], check=True, capture_output=True)
            subprocess.run(['git', 'clone', remote, str(work)], check=True, capture_output=True)
            old_cwd = os.getcwd()
            os.chdir(work)
            try:
                p.git('config', 'user.name', 'Test')
                p.git('config', 'user.email', 'test@localhost')
                Path('source.txt').write_text('source must remain unchanged')
                p.git('add', '.')
                p.git('commit', '-m', 'source')
                source_sha = p.git('rev-parse', 'HEAD')
                p.git('push', 'origin', 'HEAD:refs/heads/main')
                state = {'schema': 1, 'releases': {}}
                p.save_state(state, None)
                loaded, first = p.load_state({'baseline': {}})
                self.assertEqual(loaded, state)
                p.save_state(state, first)
                self.assertEqual(p.load_state({'baseline': {}})[1], first)
                state['releases']['1'] = {'tag': 'v1.0.0', 'status': 'built'}
                p.save_state(state, first)
                second = p.load_state({'baseline': {}})[1]
                self.assertEqual(p.git('rev-parse', second + '^'), first)
                self.assertEqual(p.git('ls-remote', 'origin', 'refs/heads/main').split()[0], source_sha)
                # A stale concurrent writer must be rejected, not overwrite the newer ledger.
                with self.assertRaises(subprocess.CalledProcessError):
                    p.save_state({'schema': 1, 'releases': {'2': {'tag': 'v2.0.0'}}}, first)
            finally:
                os.chdir(old_cwd)


if __name__ == '__main__':
    unittest.main()
