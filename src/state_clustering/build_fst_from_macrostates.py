from pyfoma import FST
from src.state_clustering.types import Macrostate


def build_fst_from_macrostates(macrostates: set[Macrostate], initial_state: Macrostate) -> FST:
    """Given a set of macrostates and some initial state, producing the appropriate FSt."""

    