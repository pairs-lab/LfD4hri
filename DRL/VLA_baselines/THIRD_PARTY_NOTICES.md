# Third-party source

Upstream model repositories are **not vendored**. `UPSTREAMS.json` records their
public URLs and the local checked-out commits; `upstream_patches/` preserves local
text changes and added source files. Fetch only the repositories you need with
`tools/restore_upstreams.py`. Each upstream retains its own license and citation
requirements; the restore tool does not grant rights to redistribute its weights
or data. Binary dependencies, weights and nested submodules are not restored.

The derived Robotiq geometry in `VLA/assets/` retains its existing attribution.
The main robot/object assets remain external in the parent DRL checkout.
No new license is assigned to the project's own code by this packaging step;
choose/confirm the project's license before publishing it publicly.

Historical regression tests can depend on omitted experiment receipts or on
excluded storage-maintenance scripts (for example the cleanup-policy assertion
in `VLA/falcon_esm_r91/tests.py`). They are retained as source, not advertised as
a dependency-free full test suite. `tools/verify_source.py` runs only source
checks and the numerical 21-D contract tests.
