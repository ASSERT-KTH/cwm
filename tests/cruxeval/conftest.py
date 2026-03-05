import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tokenizer-path",
        default=None,
        help="Path to CWMInstructTokenizer model file",
    )


@pytest.fixture
def tokenizer_path(request):
    path = request.config.getoption("--tokenizer-path", default=None)
    if not path:
        pytest.skip("tokenizer not available (pass --tokenizer-path)")
    return path
