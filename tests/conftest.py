def pytest_addoption(parser):
    parser.addoption("--run-integration", default=False, help="Run integration tests")
