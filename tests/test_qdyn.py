"""Tests for `qdyn` package."""

import pytest
from pkg_resources import parse_version

import qdyn


def test_valid_version():
    """Check that the package defines a valid __version__"""
    assert parse_version(qdyn.__version__) >= parse_version("0.3.0-dev")
