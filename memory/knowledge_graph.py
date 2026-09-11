"""Relational episodic knowledge graph with temporal multi-hop recall."""
from __future__ import annotations
import sqlite3,time
from typing import Any,Dict,Iterable,List,Optional
class RelationalKnowledgeGraph:
    def __init__(self,db_path:str):self.db_path=db_path;self._init_schema()
    def _connect(self):
        c=sqlite3.connect(self.db_path);c.row_factory=sqlite3.Row;return c
    def _init_schema(self):
        with self._connect() as c:
            c.executescript("""
CREATE TABLE IF NOT EXISTS graph_nodes(id INTEGER PRIMARY KEY AUTOINCREMENT,entity TEXT NOT NULL UNIQUE,entity_type TEXT NOT NULL DEFAULT 'concept',created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS graph_edges(id INTEGER PRIMARY KEY AUTOINCREMENT,source_entity TEXT NOT NULL,predicate TEXT NOT NULL,target_entity TEXT NOT NULL,weight REAL NOT NULL DEFAULT 1.0,temporal_session TEXT NOT NULL DEFAULT 'main',timestamp REAL NOT NULL,UNIQUE(source_entity,predicate,target_entity,temporal_session));
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_graph_edges_session ON graph_edges(temporal_session,timestamp);
CREATE TABLE IF NOT EXISTS episodic_interactions(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL,prompt TEXT NOT NULL,completion TEXT NOT NULL,reward REAL NOT NULL DEFAULT 0.0,surprise_score REAL NOT NULL DEFAULT 0.0,domain TEXT NOT NULL DEFAULT 'general',consolidated INTEGER NOT NULL DEFAULT 0,timestamp REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_epi_replay ON episodic_interactions(consolidated,reward,surprise_score DESC);
""")
    def insert_triple(self,source,predicate,target,weight=1.0,session_id="main",source_type="concept",target_type="concept"):
        now=time.time()
        with self._connect() as c:
            c.execute("INSERT OR IGNORE INTO graph_nodes(entity,entity_type,created_at) VALUES(?,?,?)",(source,source_type,now));c.execute("INSERT OR IGNORE INTO graph_nodes(entity,entity_type,created_at) VALUES(?,?,?)",(target,target_type,now));c.execute("""INSERT INTO graph_edges(source_entity,predicate,target_entity,weight,temporal_session,timestamp) VALUES(?,?,?,?,?,?) ON CONFLICT(source_entity,predicate,target_entity,temporal_session) DO UPDATE SET weight=excluded.weight,timestamp=excluded.timestamp""",(source,predicate,target,float(weight),session_id,now))
    def insert_many(self,triples):
        n=0
        for x in triples:self.insert_triple(str(x['source']),str(x['predicate']),str(x['target']),float(x.get('weight',1)),str(x.get('session_id','main')));n+=1
        return n
    def recursive_multi_hop_query(self,start_entity,max_depth=3,session_id=None):
        where="";params=[start_entity]
        if session_id is not None:where=" AND temporal_session=?";params.append(session_id)
        params.append(max(1,int(max_depth)))
        sql=f"""WITH RECURSIVE hops(source_entity,predicate,target_entity,weight,temporal_session,timestamp,depth,path) AS (SELECT source_entity,predicate,target_entity,weight,temporal_session,timestamp,1,'|'||source_entity||'|'||target_entity||'|' FROM graph_edges WHERE source_entity=? {where} UNION ALL SELECT e.source_entity,e.predicate,e.target_entity,e.weight,e.temporal_session,e.timestamp,h.depth+1,h.path||e.target_entity||'|' FROM graph_edges e JOIN hops h ON e.source_entity=h.target_entity WHERE h.depth<? AND instr(h.path,'|'||e.target_entity||'|')=0) SELECT * FROM hops ORDER BY depth ASC,weight DESC,timestamp DESC"""
        with self._connect() as c:rows=c.execute(sql,params).fetchall()
        return [dict(r) for r in rows]
    def search_entities(self,text,limit=20):
        with self._connect() as c:rows=c.execute("SELECT entity FROM graph_nodes WHERE lower(entity) LIKE ? ORDER BY entity LIMIT ?",(f"%{text.lower()}%",int(limit))).fetchall()
        return [str(r['entity']) for r in rows]
    def recall(self,query,max_depth=3,limit=50):
        entities=self.search_entities(query,min(limit,20))
        if not entities:
            needle=f"%{query.lower()}%"
            with self._connect() as c:rows=c.execute("SELECT DISTINCT source_entity FROM graph_edges WHERE lower(predicate) LIKE ? OR lower(target_entity) LIKE ? ORDER BY timestamp DESC LIMIT ?",(needle,needle,min(limit,20))).fetchall()
            entities=[str(r['source_entity']) for r in rows]
        out=[];seen=set()
        for e in entities:
            for r in self.recursive_multi_hop_query(e,max_depth):
                k=(r['source_entity'],r['predicate'],r['target_entity'],r['temporal_session'])
                if k in seen:continue
                seen.add(k);out.append(r)
                if len(out)>=limit:return out
        return out
    def stats(self):
        with self._connect() as c:
            nodes=c.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0];edges=c.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0];sessions=c.execute("SELECT COUNT(DISTINCT temporal_session) FROM graph_edges").fetchone()[0]
        return {"nodes":int(nodes),"edges":int(edges),"sessions":int(sessions)}
    def log_interaction(self,session_id,prompt,completion,reward,surprise,domain="general"):
        with self._connect() as c:cur=c.execute("INSERT INTO episodic_interactions(session_id,prompt,completion,reward,surprise_score,domain,consolidated,timestamp) VALUES(?,?,?,?,?,?,0,?)",(str(session_id),str(prompt),str(completion),float(reward),float(surprise),str(domain),time.time()));return int(cur.lastrowid)
    def fetch_unconsolidated_high_surprise(self,min_surprise,limit=32):
        with self._connect() as c:rows=c.execute("SELECT * FROM episodic_interactions WHERE consolidated=0 AND reward>=0.8 AND surprise_score>=? ORDER BY surprise_score DESC,id ASC LIMIT ?",(float(min_surprise),int(limit))).fetchall()
        return [dict(r) for r in rows]
    def mark_consolidated(self,interaction_ids):
        ids=[int(x) for x in interaction_ids if isinstance(x,(int,float))]
        if ids:
            with self._connect() as c:c.execute(f"UPDATE episodic_interactions SET consolidated=1 WHERE id IN ({','.join('?' for _ in ids)})",ids)
