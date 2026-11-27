"""Riemannian Registration / Co-Registration with nuclear-norm sign optimization."""
from .core import RCoRDetails, nuclearnorm, rr, rr_from_evd

__all__ = ["RCoRDetails", "nuclearnorm", "rr", "rr_from_evd"]
