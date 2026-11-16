"""Shape and mathematical invariant tests wrapper."""

try:
    from test_shapes_and_contracts import TestShapesAndContracts
except ImportError:
    from tests.test_shapes_and_contracts import TestShapesAndContracts

__all__ = ["TestShapesAndContracts"]
