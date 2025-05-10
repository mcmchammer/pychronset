"""Test pychronset."""

import pychronset


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(pychronset.__name__, str)
