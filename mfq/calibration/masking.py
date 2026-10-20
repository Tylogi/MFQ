"""Transformers attention-mask helpers shared by layerwise backends."""

from __future__ import annotations


def create_causal_mask_compat(factory, **arguments):
    """Call the Transformers mask API across its cache-position transition."""

    try:
        return factory(**arguments)
    except TypeError as error:
        # Transformers 5.14 removed the deprecated cache_position argument;
        # earlier supported releases still require it.
        if "cache_position" not in str(error):
            raise
        return factory(**arguments, cache_position=None)


__all__ = ["create_causal_mask_compat"]
