def pytest_addoption(parser):
    parser.addoption(
        "--generate-baseline",
        action="store_true",
        default=False,
        help="Write new regression baselines instead of comparing against existing ones.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_wrap: test requires wine64 and WRAP SIM.exe to be present on this machine",
    )
