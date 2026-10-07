from mfq._alphaq_profiles import ALPHAQ_PROFILES


CANDIDATE_GROUPS = (
    ('NVQ', tuple(name for name in ALPHAQ_PROFILES if name.startswith('NVQ'))),
    ('NINT', tuple(name for name in ALPHAQ_PROFILES if name.startswith('NINT'))),
    ('MXFP4-SQ', tuple(f'MXFP4-SQ-{q}' for q in ('1', '2', '3', 'F'))),
    ('MXFP8-SQ', tuple(f'MXFP8-SQ-{q}' for q in range(1, 9))),
    ('FP8-SQ', tuple(f'FP8-SQ-{q}' for q in range(1, 9))),
)
WORKBENCH_CANDIDATES = tuple(name for _, names in CANDIDATE_GROUPS for name in names)
SQ_CANDIDATES = frozenset(WORKBENCH_CANDIDATES) - frozenset(ALPHAQ_PROFILES)
