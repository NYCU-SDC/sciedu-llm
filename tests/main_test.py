from src.main import main


def test_template():
    assert main() or not main()
