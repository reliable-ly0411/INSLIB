#!/usr/bin/env python3
"""Release gate. Standard-library only; no publishing or credential management.

AP_FLAKE8_CLEAN
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

CONFIG = '.github/release_policy/config.json'
STATE_REF = 'refs/heads/release-policy-state'
STATE_FILE = '.release-policy-state.json'
CONTROL_PATHS = ['.github/workflows', '.github/release_policy']


def command(*args, cwd=None, env=None, data=None):
    return subprocess.check_output(args, cwd=cwd, env=env, input=data, text=True).strip()


def git(*args, **kwargs):
    return command('git', *args, **kwargs)


def family_for(tag, config):
    matches = [family for family, pattern in config['families'].items() if re.fullmatch(pattern, tag)]
    if len(matches) > 1:
        raise ValueError('Ambiguous release family: ' + tag)
    return matches[0] if matches else None


def new_state(config):
    return {'schema': 1, 'releases': copy.deepcopy(config['baseline'])}


def select_release(releases, state, config, resolve_tag, retry_id=None):
    """Return one new formal release. Existing stable releases are never backfilled.

    Retry preserves the first pinned manifest; three automatic attempts maximum.
    A prerelease observed before promotion to stable remains eligible even if its
    original publication timestamp predates bootstrap. Changed IDs/tags fail closed.
    """
    records = state['releases']
    candidates = []
    tag_ids = {entry['tag']: rid for rid, entry in records.items()}
    present_ids = {str(release['id']) for release in releases}
    for rid, entry in records.items():
        if rid not in present_ids and entry['status'] not in ('blocked', 'prerelease'):
            entry.update(status='blocked', reason='Release was deleted or is no longer visible')
    for release in releases:
        rid = str(release['id'])
        tag = release['tag_name']
        old = records.get(rid)
        if tag in tag_ids and tag_ids[tag] != rid:
            records[rid] = {'tag': tag, 'status': 'blocked', 'reason': 'Tag reused by a different release ID'}
            continue
        tag_ids[tag] = rid
        if old and old['tag'] != tag:
            old.update(status='blocked', reason='Release tag was renamed')
            continue
        if release.get('draft') or release.get('prerelease'):
            if old and old['status'] in ('building', 'failed', 'built'):
                old.update(status='blocked', reason='Formal release status was revoked')
            if not old:
                records[rid] = {'tag': tag, 'status': 'prerelease'}
            continue
        family = family_for(tag, config)
        if not family or not release.get('published_at'):
            continue
        if not old and release['published_at'] <= config['bootstrap_at']:
            records[rid] = {'tag': tag, 'status': 'baseline', 'upstream_sha': resolve_tag(tag)}
            continue
        if old and old['status'] in ('baseline', 'built', 'blocked'):
            if old['status'] != 'blocked' and old.get('upstream_sha') and resolve_tag(tag) != old['upstream_sha']:
                old.update(status='blocked', reason='Historical release tag moved or disappeared')
            continue
        if retry_id and rid != retry_id:
            continue
        if old and old.get('attempts', 0) >= config['max_attempts'] and retry_id != rid:
            continue
        candidates.append((release['published_at'], rid, family, release))
    for _, rid, family, release in sorted(candidates):
        tag = release['tag_name']
        sha = resolve_tag(tag)
        if not re.fullmatch(r'[0-9a-f]{40}', sha or ''):
            raise ValueError('Release tag did not resolve to a commit: ' + tag)
        old = records.get(rid, {})
        if old.get('upstream_sha') and old['upstream_sha'] != sha:
            old.update(status='blocked', reason='Release tag moved after recognition')
            continue
        entry = dict(old, tag=tag, family=family, upstream_sha=sha,
                     published_at=release['published_at'])
        records[rid] = entry
        return rid, entry
    return None


def load_config():
    config = json.loads(Path(CONFIG).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', config['upstream']):
        raise ValueError('Invalid upstream repository')
    return config


def load_state(config):
    refs = git('ls-remote', '--heads', 'origin', STATE_REF)
    if not refs:
        return new_state(config), None
    git('fetch', '--no-recurse-submodules', '--no-tags', 'origin', STATE_REF)
    sha = git('rev-parse', 'FETCH_HEAD')
    return json.loads(git('show', sha + ':' + STATE_FILE)), sha


def save_state(state, parent):
    """Append a commit to a dedicated branch; no force pushes, tags or workflows."""
    if parent and json.loads(git('show', parent + ':' + STATE_FILE)) == state:
        return parent
    base = parent or git('rev-parse', 'HEAD')
    data = json.dumps(state, indent=2, sort_keys=True) + '\n'
    blob = git('hash-object', '-w', '--stdin', data=data)
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / 'index'))
        git('read-tree', base, env=env)
        git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + STATE_FILE, env=env)
        tree = git('write-tree', env=env)
        commit = git('commit-tree', tree, '-p', base, data='CI: record release gate state\n')
    # Existing state ref is never rewritten. Concurrent writers fail rather than lose state.
    git('push', 'origin', commit + ':' + STATE_REF)


def resolve_tag(config, tag):
    git('check-ref-format', 'refs/tags/' + tag)
    git('fetch', '--no-recurse-submodules', '--no-tags', 'https://github.com/' + config['upstream'] + '.git',
        'refs/tags/' + tag)
    return git('rev-parse', 'FETCH_HEAD^{commit}')


def output(name, value):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as handle:
        handle.write(name + '=' + str(value) + '\n')


def prepare(retry_id=None):
    config = load_config()
    state, parent = load_state(config)
    pages = json.loads(command('gh', 'api', '--paginate', '--slurp',
                               'repos/' + config['upstream'] + '/releases?per_page=100'))
    releases = [release for page in pages for release in page]
    if retry_id and state['releases'].get(retry_id, {}).get('status') not in ('failed', 'building'):
        raise ValueError('Only a failed/incomplete recognized release may be explicitly retried')
    refs = git('ls-remote', '--tags', 'https://github.com/' + config['upstream'] + '.git')
    tags = {}
    for line in refs.splitlines():
        sha, ref = line.split()
        tag = ref.removeprefix('refs/tags/').removesuffix('^{}')
        if ref.endswith('^{}') or tag not in tags:
            tags[tag] = sha
    selected = select_release(releases, state, config, tags.get, retry_id)
    output('build', 'false')
    if not selected:
        save_state(state, parent)
        return
    rid, entry = selected
    if resolve_tag(config, entry['tag']) != entry['upstream_sha']:
        raise ValueError('Release tag changed during source recognition')
    if 'manifest' not in entry:
        downstream = git('rev-parse', 'HEAD')
        git('fetch', '--no-recurse-submodules', '--no-tags',
            'https://github.com/' + config['upstream'] + '.git', config['branch'])
        merge_base = git('merge-base', downstream, 'FETCH_HEAD')
        manifest = {'schema': 1, 'release_id': rid, 'tag': entry['tag'], 'family': entry['family'],
                    'upstream': config['upstream'], 'upstream_sha': entry['upstream_sha'],
                    'downstream_sha': downstream, 'patch_base': merge_base,
                    'published_at': entry['published_at'], 'publication': config['publication']}
        manifest['dependencies'] = {}
        for name, dependency in config.get('dependencies', {}).items():
            refs = git('ls-remote', 'https://github.com/' + dependency['repository'] + '.git',
                       'refs/heads/' + dependency['ref']).split()
            if len(refs) != 2 or not re.fullmatch(r'[0-9a-f]{40}', refs[0]):
                raise ValueError('Cannot pin dependency: ' + name)
            manifest['dependencies'][name] = refs[0]
        manifest['patch_sha256'] = hashlib.sha256(patch_bytes(manifest)).hexdigest()
        # Pin the deterministic source tree and commit before running any native code.
        with tempfile.TemporaryDirectory() as tmp:
            work = str(Path(tmp) / 'source')
            git('worktree', 'add', '--detach', work, downstream)
            try:
                manifest['source_sha'] = materialize(manifest, work, update_submodules=False)
            finally:
                git('worktree', 'remove', '--force', work)
        entry['manifest'] = manifest
    entry['status'] = 'building'
    entry['attempts'] = entry.get('attempts', 0) + 1
    entry['run_id'] = os.environ['GITHUB_RUN_ID']
    save_state(state, parent)
    output('build', 'true')
    output('release_id', rid)
    output('source_ref', entry['manifest']['downstream_sha'])
    output('manifest', json.dumps(entry['manifest'], separators=(',', ':')))


def patch_bytes(manifest, cwd=None):
    args = ['git', 'diff', '--binary', manifest['patch_base'], manifest['downstream_sha'], '--', '.',
            *[':(exclude)' + path + '/**' for path in CONTROL_PATHS]]
    return subprocess.check_output(args, cwd=cwd)


def materialize(manifest, cwd=None, update_submodules=True):
    """Build release tag + exact downstream delta, never current upstream HEAD."""
    for key in ('upstream_sha', 'downstream_sha', 'patch_base'):
        if not re.fullmatch(r'[0-9a-f]{40}', manifest[key]):
            raise ValueError('Invalid pinned SHA: ' + key)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', manifest['upstream']):
        raise ValueError('Invalid upstream repository')
    git('check-ref-format', 'refs/tags/' + manifest['tag'], cwd=cwd)
    git('fetch', '--no-recurse-submodules', '--no-tags', 'https://github.com/' + manifest['upstream'] + '.git',
        'refs/tags/' + manifest['tag'], cwd=cwd)
    if git('rev-parse', 'FETCH_HEAD^{commit}', cwd=cwd) != manifest['upstream_sha']:
        raise ValueError('Upstream release tag moved; refusing build')
    patch = patch_bytes(manifest, cwd)
    if hashlib.sha256(patch).hexdigest() != manifest['patch_sha256']:
        raise ValueError('Downstream patch digest mismatch')
    git('checkout', '--detach', manifest['upstream_sha'], cwd=cwd)
    if patch:
        subprocess.run(['git', 'apply', '--3way', '--index', '-'], cwd=cwd, input=patch, check=True)
    # Source builds keep reviewed downstream workflow policy, including intentional removals.
    for path in CONTROL_PATHS:
        git('rm', '-r', '--ignore-unmatch', '--', path, cwd=cwd)
        git('restore', '--source=' + manifest['downstream_sha'], '--staged', '--worktree', '--', path, cwd=cwd)
    tree = git('write-tree', cwd=cwd)
    env = dict(os.environ, GIT_AUTHOR_NAME='release-policy', GIT_AUTHOR_EMAIL='release-policy@localhost',
               GIT_COMMITTER_NAME='release-policy', GIT_COMMITTER_EMAIL='release-policy@localhost',
               GIT_AUTHOR_DATE=manifest['published_at'], GIT_COMMITTER_DATE=manifest['published_at'])
    message = 'Release candidate ' + manifest['tag'] + '\nDownstream baseline: ' + manifest['downstream_sha'] + '\n'
    commit = git('commit-tree', tree, '-p', manifest['upstream_sha'], data=message, env=env, cwd=cwd)
    if manifest.get('source_sha') and commit != manifest['source_sha']:
        raise ValueError('Reconstructed source commit differs from approved manifest')
    git('checkout', '--detach', commit, cwd=cwd)
    if update_submodules and Path(cwd or '.', '.gitmodules').exists():
        git('submodule', 'sync', '--recursive', cwd=cwd)
        git('submodule', 'update', '--init', '--recursive', cwd=cwd)
    return commit


def finish(release_id, result):
    config = load_config()
    state, parent = load_state(config)
    entry = state['releases'][release_id]
    if entry.get('run_id') != os.environ['GITHUB_RUN_ID']:
        raise ValueError('Refusing to finish a release claimed by another run')
    entry['status'] = 'built' if result == 'success' else 'failed'
    entry['result'] = result
    entry['publication'] = config['publication']
    save_state(state, parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'materialize', 'finish'])
    parser.add_argument('--release-id')
    parser.add_argument('--result')
    parser.add_argument('--retry-id')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.retry_id)
    elif args.action == 'finish':
        finish(args.release_id, args.result)
    else:
        raw = os.environ.get('RELEASE_MANIFEST', '')
        if raw:
            manifest = json.loads(raw)
            materialize(manifest)
            path = Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir())) / 'release-source.json'
            path.write_text(json.dumps(manifest, indent=2) + '\n')
            print('Verified release source:', manifest['source_sha'])
        else:
            print('Explicit manual build of selected ref; no automatic publication.')


if __name__ == '__main__':
    main()
