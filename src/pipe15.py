"""v28: v18 matcher + knowledge-based rewriting of hangul names (meaning words, Korean company names).
Rows without such tokens produce exactly the same query as v18."""
from __future__ import annotations

from pipe11 import *  # noqa: F401,F403
from pipe11 import Matcher11
import alias_ko


class Matcher15(Matcher11):
    def query_info(self, row: dict):
        q = super().query_info(row)
        if not q["hang"]:
            return q
        for v in alias_ko.rewrite(q["name"]):
            qv = super().query_info(dict(row, vessel_name=v))
            if not qv["hang"]:
                continue
            for key in ("sk", "vk", "units", "romv", "units_red"):
                seen = set(map(repr, q[key]))
                for item in qv[key]:
                    if repr(item) not in seen:
                        q[key].append(item); seen.add(repr(item))
        return q
