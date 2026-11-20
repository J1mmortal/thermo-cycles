# Releasing thermo-cycles

Publishing is automated: pushing a tag `vX.Y.Z` runs `.github/workflows/release.yml`. It builds the sdist and
wheel, runs `twine check --strict`, installs the wheel into a clean venv and runs the test suite and the examples
against it, checks that the tag equals `thermo_cycles.__version__`, and publishes the **same files** to PyPI. It
uses PyPI Trusted Publishing (OpenID Connect), so **no API token or password is stored anywhere**.

## One-time setup (Jesper)

1. **Accounts.** Create accounts on <https://pypi.org> and <https://test.pypi.org> (they are separate). Enable 2FA
   on both.
2. **Pending trusted publisher on PyPI.** The project name does not exist yet, so use a *pending* publisher
   (PyPI: Account settings, Publishing, "Add a new pending publisher"). Fill in:
   - PyPI project name: `thermo-cycles`
   - Owner: `J1mmortal`
   - Repository name: `thermo-cycles`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   A pending publisher does not reserve the name, so register it shortly before the first release. The first
   successful publish creates the project and turns the pending publisher into a normal one.
3. **Same on TestPyPI** (optional, recommended as a dry run): a pending publisher at
   <https://test.pypi.org/manage/account/publishing/> with the same owner, repository and workflow, and
   environment name `testpypi`.
4. **GitHub environments.** In the repository: Settings, Environments, create `pypi` and `testpypi`. Suggested
   protection for `pypi`: "Required reviewers" (yourself), so a tag push waits for your approval before anything is
   uploaded, and "Deployment branches and tags" limited to tags matching `v*`.
5. **Dry run (optional).** Actions tab, workflow "release", "Run workflow": this builds, tests the wheel and
   publishes to TestPyPI only. A version can be uploaded to TestPyPI only once; if a dry run fails halfway, bump
   the version to retry.

## Per release

1. Branch from `main`. Bump `__version__` in `src/thermo_cycles/__init__.py` (the only place; `pyproject.toml`
   reads it from there) and the version mention in the first paragraph of the README. The test
   `test_import_and_version` pins the number; update it too.
2. Move the entries in `CHANGELOG.md` under the new version and date; update the compare links at the bottom.
3. Run locally: `python -m pytest -q`, then `python -m build` and `python -m twine check dist/*`.
4. Merge to `main` through a PR (the `tests` workflow must be green).
5. Tag the merge commit and push the tag:
   ```
   git checkout main
   git pull
   git tag -a vX.Y.Z -m "thermo-cycles X.Y.Z"
   git push origin vX.Y.Z
   ```
6. Watch the Actions run. If the `pypi` environment has required reviewers, approve the deployment.
7. Check <https://pypi.org/project/thermo-cycles/>: version, README rendering, licence, classifiers, project links.
   Then in a fresh venv: `pip install "thermo-cycles[plot]"` and
   `python -c "import thermo_cycles as t; print(t.__version__)"`.
8. Optionally create a GitHub release from the tag with the changelog entry.

A version can never be re-uploaded to PyPI, even after deleting it. If a release is bad, yank it on PyPI and publish
a new patch version. If the tag and the version differ, the workflow fails before publishing: delete the tag
(`git push origin :refs/tags/vX.Y.Z`), fix, and tag again.

## References

- PyPI Trusted Publishers: <https://docs.pypi.org/trusted-publishers/>
- Pending publishers (project not yet on PyPI): <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>
- Using a publisher from GitHub Actions: <https://docs.pypi.org/trusted-publishers/using-a-publisher/>
- Python Packaging User Guide, publishing with GitHub Actions: <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>
