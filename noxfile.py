import nox

PYPROJECT = nox.project.load_toml("pyproject.toml")
PYTHON_VERSIONS = nox.project.python_versions(PYPROJECT)
NUMPY = {
    "1.22": {"3.10"},
    "2.0": {"3.10", "3.11", "3.12"},
    "2.5": {"3.12", "3.13", "3.14"},
}


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("numpy", list(NUMPY))
def tests(session, numpy):
    if session.python not in NUMPY[numpy]:
        session.skip("Unsupported numpy version")
    session.install(".", "--group", "tests", f"numpy=={numpy}")
    session.run("pytest", "--no-cov", "--inline-snapshot=disable")
