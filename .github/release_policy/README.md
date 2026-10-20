# Release-gated downstream automation

AI-assisted policy change. Native build/test bodies and existing security/owner gates are retained.

- Scheduled Sync upstream performs a non-destructive merge, preserving downstream workflow policy, then checks formal upstream releases. Ordinary pushes/PRs run only the policy checks and retained lightweight repository checks.
- The initial snapshot marks every existing stable release as seen. Drafts/prereleases do not build. A later promotion of an observed prerelease to stable is recognized. Families are tracked independently in config.json; one eligible release is processed per scheduled run.
- Source is the peeled release tag plus the downstream-only patch against the synchronized upstream merge-base. The downstream baseline, patch hash, source commit and dependency pins are immutable in the manifest. Current unreleased upstream HEAD is not built as a release.
- The release-policy-state branch is an append-only operational ledger. `building` and `failed` are retryable up to three automatic attempts, retaining the same manifest; `built` is deduplicated. Moved/renamed tags are blocked. Retry a failed ID explicitly with Sync upstream / check-releases / retry_release_id. A successful build is never an authorization to publish.
- Manual sync defaults to sync-only. Native workflows retain manual workflow_dispatch interfaces. Automatically callable recipes additionally accept source_ref; blank means the selected workflow ref. Other inherited manual matrices use the selected workflow ref. These explicit manual builds bypass release eligibility but not quality, owner or security gates.
- Release candidate validation runs the automatic_workflows listed in config.json. Other retained native/platform matrices are manual-only, not removed. Each automatically called checkout reconstructs and checks the exact manifest source.
- The Promote verified artifacts interface verifies a completed candidate run and its immutable artifact digest without rebuilding. Publication remains blocked until an authorized downstream destination and original human/security gates are available. No tags, releases, credentials, repository settings or external storage are created or overwritten by this policy.
- The release evidence artifact records source pins, workflow run and artifact IDs/digests. It is evidence, not permission to fly, flash or publish firmware. Murmur public test-key firmware is never uploaded or eligible for device publication.

Validation: run `python3 -m unittest discover -s .github/release_policy -p 'test_*.py' -v` and inspect the policy-checks workflow. Full native matrices run only through a qualifying release or an explicit manual action. Existing running jobs are not canceled by this change.

## Current publication limits

The automatic gate runs existing CI quality jobs only. The Windows GUI and PyPI release workflows retain the `jnz/INSLIB` owner check. No fork PyPI/TestPyPI identity, trusted publisher, Pages deployment or signing credential is created or enabled. Their original manual entry points remain subject to that owner gate.

This implements release-gated candidate validation, not a complete automatic public release pipeline. A new upstream formal release is necessary but not sufficient for publication; unsupported destinations, signing/identity requirements and human gates remain blocking.
