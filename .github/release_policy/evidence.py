#!/usr/bin/env python3
"""Record/verify candidate artifact provenance without rebuilding or publishing.

AP_FLAKE8_CLEAN
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], text=True))


def artifacts(repo, run_id):
    pages = json.loads(subprocess.check_output(
        ['gh', 'api', '--paginate', '--slurp',
         f'repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100'], text=True))
    return [artifact for page in pages for artifact in page['artifacts']]


def validate_digest(artifact):
    digest = artifact.get('digest', '')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Artifact lacks a verifiable immutable digest: ' + artifact['name'])
    if artifact.get('expired'):
        raise ValueError('Artifact expired; do not silently rebuild for promotion: ' + artifact['name'])
    return digest


def archive(repo, artifact):
    expected = validate_digest(artifact)
    data = subprocess.check_output(['gh', 'api', f'repos/{repo}/actions/artifacts/{artifact["id"]}/zip'])
    actual = 'sha256:' + hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError('Artifact archive digest mismatch: ' + artifact['name'])
    return data


def verify_upstream(manifest):
    upstream = manifest['upstream']
    release_id = manifest['release_id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', upstream):
        raise ValueError('Invalid upstream repository in evidence')
    if not re.fullmatch(r'[0-9]+', str(release_id)):
        raise ValueError('Invalid upstream release ID in evidence')
    release = api(f'repos/{upstream}/releases/{release_id}')
    if release['draft'] or release['prerelease'] or release['tag_name'] != manifest['tag']:
        raise ValueError('Upstream release is no longer formal or its tag was renamed')
    refs = subprocess.check_output(
        ['git', 'ls-remote', '--tags', 'https://github.com/' + upstream + '.git',
         'refs/tags/' + manifest['tag'], 'refs/tags/' + manifest['tag'] + '^{}'], text=True)
    sha = None
    for line in refs.splitlines():
        current, ref = line.split()
        if sha is None or ref.endswith('^{}'):
            sha = current
    if sha != manifest['upstream_sha']:
        raise ValueError('Upstream release tag moved or disappeared after candidate build')


def record():
    repo = os.environ['GITHUB_REPOSITORY']
    run_id = os.environ['GITHUB_RUN_ID']
    manifest = json.loads(os.environ['RELEASE_MANIFEST'])
    files = artifacts(repo, run_id)
    for artifact in files:
        validate_digest(artifact)
    evidence = {'schema': 1, 'repository': repo, 'run_id': run_id,
                'classification': 'validation-only; not approved for device use or publication',
                'manifest': manifest,
                'artifacts': [{key: a[key] for key in ('id', 'name', 'digest', 'size_in_bytes')} for a in files]}
    Path('release-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')


def verify():
    repo = os.environ['GITHUB_REPOSITORY']
    run_id = os.environ['CANDIDATE_RUN_ID']
    if not re.fullmatch(r'[0-9]+', run_id):
        raise ValueError('Candidate run ID must contain digits only')
    run = api(f'repos/{repo}/actions/runs/{run_id}')
    if run['conclusion'] != 'success' or run['path'] != '.github/workflows/sync-upstream.yml':
        raise ValueError('Only a successful release-gated sync candidate can be promoted')
    files = artifacts(repo, run_id)
    matches = [a for a in files if a['name'] == 'release-evidence']
    if len(matches) != 1:
        raise ValueError('Expected exactly one immutable release-evidence artifact')
    with zipfile.ZipFile(io.BytesIO(archive(repo, matches[0]))) as source:
        evidence = json.loads(source.read('release-evidence.json'))
    if evidence['repository'] != repo or evidence['run_id'] != run_id:
        raise ValueError('Evidence does not belong to the requested repository/run')
    content = api(f'repos/{repo}/contents/.release-policy-state.json?ref=release-policy-state')
    state = json.loads(base64.b64decode(content['content']))
    entry = state['releases'][evidence['manifest']['release_id']]
    if entry['status'] != 'built' or entry['run_id'] != run_id or entry['manifest'] != evidence['manifest']:
        raise ValueError('Candidate is not the successful immutable manifest in the release ledger')
    verify_upstream(evidence['manifest'])
    by_id = {a['id']: a for a in files}
    for recorded in evidence['artifacts']:
        current = by_id.get(recorded['id'])
        if not current or current['digest'] != recorded['digest'] or current['name'] != recorded['name']:
            raise ValueError('Recorded artifact is absent or metadata differs')
        archive(repo, current)
    print('Verified source manifest and', len(evidence['artifacts']), 'artifact digests without rebuilding.')
    print('Publication remains blocked:', entry['publication'])


if __name__ == '__main__':
    {'record': record, 'verify': verify}[sys.argv[1]]()
