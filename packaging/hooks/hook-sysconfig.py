"""Remove build-interpreter paths from frozen Python configuration metadata."""

import sys
import sysconfig
from types import CodeType


def replace_build_prefix(value, prefix):
    if isinstance(value, CodeType):
        return value.replace(co_consts=tuple(replace_build_prefix(item, prefix) for item in value.co_consts))
    if isinstance(value, str):
        return value.replace(prefix, "/mfq-build/python")
    if isinstance(value, tuple):
        return tuple(replace_build_prefix(item, prefix) for item in value)
    if isinstance(value, frozenset):
        return frozenset(replace_build_prefix(item, prefix) for item in value)
    return value


def hook(hook_api):
    name = sysconfig._get_sysconfigdata_name()
    for node in hook_api.module_graph.import_hook(name, hook_api.module):
        if isinstance(node.code, CodeType):
            node.code = replace_build_prefix(node.code, sys.base_prefix)
