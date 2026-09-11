"""Episodic DB compatibility plus truthful reward semantics for unverified chat."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import List, Optional
from memory._db_base import *
from memory import _db_base as _base

class EpisodicMemoryDB(_base.EpisodicMemoryDB):
    def log_interaction(self,prompt:str,completion:str,raw_branches=None,verified_reward:Optional[float]=None,surprise_score:float=0.0,mode:str="Instant",entropy:float=0.0,winning_branch:int=0,test_cases:Optional[str]=None,winning_temp:float=0.20,**kwargs)->int:
        if "winning_temp" in kwargs:winning_temp=kwargs["winning_temp"]
        has_tests=bool(test_cases and str(test_cases).strip())
        if not has_tests and ("RLVR" not in str(mode).upper()):verified_reward=None
        with self._get_connection() as conn:
            cur=conn.cursor();cur.execute("""
                INSERT INTO interactions(prompt,completion,raw_branches,verified_reward,surprise_score,mode,entropy,winning_branch,winning_temp,test_cases,consolidated,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,0,?)
            """,(prompt,completion,json.dumps(raw_branches) if raw_branches else None,verified_reward,float(surprise_score),mode,float(entropy),int(winning_branch),float(winning_temp),test_cases,datetime.now(timezone.utc).isoformat()))
            conn.commit();return int(cur.lastrowid)
