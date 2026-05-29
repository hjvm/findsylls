from typing import Union, Optional
from .base import BaseSAD
from .energy import EnergyVAD

try:
    from .silero import SileroVAD
except ImportError:
    SileroVAD = None  # type: ignore[assignment,misc]


_SAD_REGISTRY = {
    "energy": EnergyVAD,
    "silero": SileroVAD,
}
_SAD_ALIASES = {
    "default": "energy",
    "energy_vad": "energy",
    "silero_vad": "silero",
}


def resolve_sad(sad: Union[None, bool, str, "BaseSAD"]) -> Optional["BaseSAD"]:
    """Resolve a SAD specifier to a BaseSAD instance.

    Accepts:
        None         — no SAD (pass-through)
        BaseSAD      — use directly
        True         — EnergyVAD() with defaults (the no-dependency default)
        "default"    — same as True
        "energy"     — EnergyVAD() with defaults
        "silero"     — SileroVAD() with defaults (requires torch)

    Returns:
        BaseSAD instance or None.
    """
    if sad is None or isinstance(sad, BaseSAD):
        return sad
    if sad is True or sad == "default":
        return EnergyVAD()
    if isinstance(sad, str):
        key = _SAD_ALIASES.get(sad, sad)
        cls = _SAD_REGISTRY.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown SAD name {sad!r}. "
                f"Valid names: {sorted(_SAD_REGISTRY)} (plus aliases: {sorted(_SAD_ALIASES)})."
            )
        return cls()
    raise TypeError(
        f"sad must be None, True, a string name, or a BaseSAD instance; got {type(sad).__name__!r}."
    )


__all__ = ["BaseSAD", "EnergyVAD", "SileroVAD", "resolve_sad"]
