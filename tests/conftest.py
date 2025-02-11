import pytest
from pytest_allclose import report_rmses


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)


def pytest_addoption(parser):
    parser.addoption("--cook", action="store_true", default=False, \
                                help="run cook tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "cook: mark test as cook.")


def pytest_collection_modifyitems(config, items):
    has_cook_opts = config.getoption("--cook")
    skip_cook = pytest.mark.skip(reason="need --cook option to run")
    skip_nocook = pytest.mark.skip(reason="cannot run if --cook is set")

    for item in items:
        if "cook" in item.keywords and not has_cook_opts:
            item.add_marker(skip_cook)
        elif "cook" not in item.keywords and has_cook_opts:
            item.add_marker(skip_nocook)
