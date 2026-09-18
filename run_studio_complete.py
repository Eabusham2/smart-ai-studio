"""Compatibility runtime for the 4K evaluation harness and OGP monitor.

The desktop product uses app_gui.py + core/*. This module keeps benchmark-only
legacy names in one place without duplicating the full application stack.
"""
from __future__ import annotations
import json, math, os, platform, re, shutil, sqlite3, subprocess, sys, tempfile, threading, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import psutil
try:
    import resource
    HAS_RESOURCE=True
except ImportError:
    resource=None; HAS_RESOURCE=False
MLX_AVAILABLE=False
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tuner.lora import LoRALinear
    MLX_AVAILABLE=True
except ImportError:
    mx=nn=optim=load=make_prompt_cache=LoRALinear=None
from consolidation.projected_daemon import GramSchmidtOGPProjector, ProjectedSleepConsolidationDaemon
METAL_STREAM_LOCK=threading.Lock()

def compute_auto_kv_budget(total_ram_gb: float)->int:
    return 1024 if total_ram_gb<=8 else 2048 if total_ram_gb<=16 else 4096 if total_ram_gb<=32 else 8192

@dataclass
class EngineSettings:
    total_ram_gb: float=field(default_factory=lambda: psutil.virtual_memory().total/(1024**3))
    mlx_model_path: str="penkia/TernaryQuench-Qwen3.8-27B-MLX"
    max_kv_tokens: int=field(init=False)
    h2o_sink_tokens:int=4; h2o_heavy_tokens:int=64; h2o_max_budget:int=128
    lora_rank:int=4; lora_alpha:float=8.0; lora_chunk_size:int=6; total_layers:int=60
    base_learning_rate:float=1e-4; ogp_ortho_tolerance:float=1e-5
    polling_interval_seconds:float=300.0; min_surprise_threshold:float=.85; min_batch_queue_size:int=5
    sandbox_timeout_seconds:float=4.0; sandbox_max_memory_mb:int=512; enable_awake_ogp_daemon:bool=False
    db_path:str=field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","memory.db"))
    def __post_init__(self): self.max_kv_tokens=compute_auto_kv_budget(self.total_ram_gb)

class RelationalKnowledgeGraph:
    def __init__(self,db_path): self.db_path=db_path; os.makedirs(os.path.dirname(db_path) or '.',exist_ok=True); self._init_db()
    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.executescript("""
CREATE TABLE IF NOT EXISTS graph_nodes(id INTEGER PRIMARY KEY AUTOINCREMENT,entity TEXT UNIQUE NOT NULL,entity_type TEXT NOT NULL,created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS graph_edges(id INTEGER PRIMARY KEY AUTOINCREMENT,source_entity TEXT NOT NULL,predicate TEXT NOT NULL,target_entity TEXT NOT NULL,weight REAL NOT NULL,temporal_session TEXT NOT NULL,timestamp REAL NOT NULL,UNIQUE(source_entity,predicate,target_entity));
CREATE TABLE IF NOT EXISTS episodic_interactions(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL,prompt TEXT NOT NULL,completion TEXT NOT NULL,reward REAL NOT NULL,surprise_score REAL NOT NULL,domain TEXT NOT NULL,consolidated INTEGER DEFAULT 0,timestamp REAL NOT NULL);""")
    def insert_triple(self,source,predicate,target,weight=1.0,session_id="main"):
        now=time.time()
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT OR IGNORE INTO graph_nodes(entity,entity_type,created_at) VALUES(?,'concept',?)",(source,now)); c.execute("INSERT OR IGNORE INTO graph_nodes(entity,entity_type,created_at) VALUES(?,'concept',?)",(target,now)); c.execute("INSERT OR REPLACE INTO graph_edges(source_entity,predicate,target_entity,weight,temporal_session,timestamp) VALUES(?,?,?,?,?,?)",(source,predicate,target,float(weight),session_id,now))
    def recursive_multi_hop_query(self,start_entity,max_depth=3):
        sql="""WITH RECURSIVE hops AS (SELECT source_entity,predicate,target_entity,1 depth,source_entity||' -> '||predicate||' -> '||target_entity path FROM graph_edges WHERE source_entity=? UNION ALL SELECT e.source_entity,e.predicate,e.target_entity,h.depth+1,h.path||' -> '||e.predicate||' -> '||e.target_entity FROM graph_edges e JOIN hops h ON e.source_entity=h.target_entity WHERE h.depth<? AND INSTR(h.path,e.target_entity)=0) SELECT source_entity,predicate,target_entity,depth,path FROM hops"""
        with sqlite3.connect(self.db_path) as c: rows=c.execute(sql,(start_entity,max_depth)).fetchall()
        return [{"source":r[0],"predicate":r[1],"target":r[2],"depth":r[3],"path":r[4]} for r in rows]
    def log_interaction(self,session_id,prompt,completion,reward,surprise,domain="general"):
        with sqlite3.connect(self.db_path) as c: c.execute("INSERT INTO episodic_interactions(session_id,prompt,completion,reward,surprise_score,domain,timestamp) VALUES(?,?,?,?,?,?,?)",(session_id,prompt,completion,reward,surprise,domain,time.time()))
    def fetch_unconsolidated_high_surprise(self,min_surprise,limit=32):
        with sqlite3.connect(self.db_path) as c:
            c.row_factory=sqlite3.Row; return [dict(r) for r in c.execute("SELECT * FROM episodic_interactions WHERE consolidated=0 AND reward>=.8 AND surprise_score>=? ORDER BY surprise_score DESC LIMIT ?",(min_surprise,limit)).fetchall()]
    def mark_consolidated(self,ids):
        if ids:
            with sqlite3.connect(self.db_path) as c: c.execute(f"UPDATE episodic_interactions SET consolidated=1 WHERE id IN ({','.join('?' for _ in ids)})",ids)

@dataclass
class SandboxResult:
    passed:bool; execution_time_ms:float; output:str; error:Optional[str]=None; reward:float=0.0
class POSIXHardenedSandbox:
    def __init__(self,timeout_sec=4.0,max_memory_mb=512): self.timeout=timeout_sec; self.max_memory_mb=max_memory_mb
    def _limits(self):
        if HAS_RESOURCE and platform.system()!='Windows':
            try:
                m=self.max_memory_mb*1024*1024; resource.setrlimit(resource.RLIMIT_AS,(m,m)); resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(self.timeout)+1),)*2); resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
            except Exception: pass
    def execute_python_code(self,code,tests):
        t=time.perf_counter(); p=None
        try:
            with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False,encoding='utf-8') as f: p=f.name; f.write(f"import sys, math, json, collections, itertools\n{code}\n{tests}\n")
            r=subprocess.run([sys.executable,p],capture_output=True,text=True,timeout=self.timeout,preexec_fn=self._limits if HAS_RESOURCE and platform.system()!='Windows' else None)
            return SandboxResult(r.returncode==0,(time.perf_counter()-t)*1000,r.stdout,None if r.returncode==0 else r.stderr[:240],1.0 if r.returncode==0 else 0.0)
        except subprocess.TimeoutExpired: return SandboxResult(False,(time.perf_counter()-t)*1000,'','ProcessTimeout',0.0)
        except Exception as e: return SandboxResult(False,(time.perf_counter()-t)*1000,'',str(e),0.0)
        finally:
            if p and os.path.exists(p): os.remove(p)
    def verify_git_diff_patch(self,repo_structure,patch_text,test_cmd):
        t=time.perf_counter(); d=tempfile.mkdtemp(prefix='swe_jail_')
        try:
            for rel,content in repo_structure.items(): p=os.path.join(d,rel); os.makedirs(os.path.dirname(p),exist_ok=True); open(p,'w',encoding='utf-8').write(content)
            blocks=re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```",patch_text,re.I); patch=blocks[-1].strip() if blocks else patch_text.strip(); open(os.path.join(d,'task.patch'),'w').write(patch+'\n')
            ap=subprocess.run(['patch','-p1','-i','task.patch'],cwd=d,capture_output=True,text=True,timeout=2)
            if ap.returncode: return SandboxResult(False,(time.perf_counter()-t)*1000,ap.stdout,ap.stderr[:240],0.0)
            env=dict(os.environ); env['PYTHONPATH']=d+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            r=subprocess.run(test_cmd,shell=True,cwd=d,env=env,capture_output=True,text=True,timeout=self.timeout,preexec_fn=self._limits if HAS_RESOURCE and platform.system()!='Windows' else None)
            return SandboxResult(r.returncode==0,(time.perf_counter()-t)*1000,r.stdout,None if r.returncode==0 else r.stderr[:240],1.0 if r.returncode==0 else 0.0)
        except Exception as e: return SandboxResult(False,(time.perf_counter()-t)*1000,'',str(e),0.0)
        finally: shutil.rmtree(d,ignore_errors=True)
    @staticmethod
    def evaluate_dsl_expression(expr):
        m=re.search(r"\[([0-9,\s\-]+)\]\s*>>~fold\((\d+)\)\s*<#>scale\((\d+)\)",expr)
        if m:
            a=[int(x.strip()) for x in m.group(1).split(',') if x.strip()]; k=int(m.group(2))%len(a) if a else 0; s=int(m.group(3)); return [x*s for x in a[k:]+a[:k]]
        m=re.search(r"\[([0-9,\s\-]+)\]\s*@fuse\s*\[([0-9,\s\-]+)\]",expr)
        if m: return [x+y for x,y in zip([int(v) for v in m.group(1).split(',')],[int(v) for v in m.group(2).split(',')])]
        return None
class FastMCPDispatcher:
    def __init__(self,sandbox,kg): self.sandbox=sandbox; self.kg=kg
    def handle_json_rpc(self,request_json):
        try:
            r=json.loads(request_json); p=r.get('params',{}); name=p.get('name'); a=p.get('arguments',{})
            if r.get('method')=='tools/call' and name=='dsl_evaluate': out=json.dumps(self.sandbox.evaluate_dsl_expression(a.get('expression',''))); return json.dumps({'jsonrpc':'2.0','id':r.get('id'),'result':{'content':[{'type':'text','text':out}]}})
            return json.dumps({'jsonrpc':'2.0','id':r.get('id'),'error':{'code':-32601,'message':'Tool not found'}})
        except Exception as e: return json.dumps({'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':str(e)}})
class NeuromorphicLIFController:
    def __init__(self,v_thresh=.55,beta=.85): self.v_thresh=v_thresh; self.beta=beta; self.v_mem=0.; self.spike_history=[]
    def compute_convex_temperature_ladder(self,entropy,t_min=.2,t_max=.88):
        self.v_mem=self.beta*self.v_mem+(1-self.beta)*entropy; spike=int(self.v_mem>=self.v_thresh); self.v_mem=0 if spike else self.v_mem; self.spike_history.append(spike); n=4 if spike or entropy>=.65 else 2 if entropy>=.3 else 1; return n,[round(t_min*((t_max/t_min)**(i/max(1,n-1))),3) for i in range(n)] if n>1 else [0.0],spike
class ASTPrefixTrieDrafter:
    def find_draft_tokens(self,history,tokenizer=None,max_draft=4):
        if len(history)>=6:
            target=history[-3:]
            for i in range(len(history)-4,-1,-1):
                if history[i:i+3]==target: return history[i+3:i+3+max_draft]
        return []
class H2OKVCacheArena:
    def __init__(self,sink_size=4,heavy_size=64,max_budget=128): self.sink_size=sink_size; self.heavy_size=heavy_size; self.max_budget=max_budget; self.accumulated_attention_scores=[]
    def register_step_attention(self,w):
        for i,x in enumerate(w): self.accumulated_attention_scores[i:i+1]=[self.accumulated_attention_scores[i]+float(x) if i<len(self.accumulated_attention_scores) else float(x)]
    def compute_compacted_indices(self,n):
        return list(range(max(0,int(n))))
@dataclass
class MCTSNode:
    state_expression:str; parent:Any=None; children:dict=field(default_factory=dict); visits:int=0; reward:float=0.
class SymbolicMCTSSearchEngine:
    def __init__(self,sandbox,c_explore=1.414,max_simulations=32): self.sandbox=sandbox; self.max_simulations=max_simulations
    def search_best_invariant(self,expr):
        actions=['>>~fold(1) <#>scale(2)','>>~fold(2) <#>scale(3)','@fuse [1, 1, 1, 1]']; best=expr; q=0.; visits=0
        for a in actions:
            e=f'{expr} {a}'; r=self.sandbox.evaluate_dsl_expression(e); score=1.0 if isinstance(r,list) and r and any(x>0 for x in r) else .0
            if score>=q: best,q=e,score
            visits+=max(1,self.max_simulations//len(actions))
        return best,q,visits
class GRPOTrainingEngine:
    def __init__(self,model,tokenizer,sandbox,group_size=4,clip_eps=.2): self.model=model; self.tokenizer=tokenizer; self.sandbox=sandbox
    def compute_group_advantages(self,rewards):
        if not rewards:return []
        m=sum(rewards)/len(rewards); sd=math.sqrt(sum((r-m)**2 for r in rewards)/max(1,len(rewards)-1))+1e-8; return [(r-m)/sd for r in rewards]
class StagedRoundRobinLoRATrainer:
    def __init__(self,model,total_layers=60,chunk_size=6): self.model=model; self.total_layers=total_layers; self.chunk_size=chunk_size
    def unfreeze_active_chunk(self,chunk_idx):
        layers=getattr(self.model,'layers',[]) or getattr(getattr(self.model,'model',None),'layers',[]); a=chunk_idx*self.chunk_size; b=min(a+self.chunk_size,len(layers))
        for i,l in enumerate(layers): getattr(l,'unfreeze' if a<=i<b else 'freeze',lambda:None)()
class HierarchicalMoERouter:
    def __init__(self,model): self.model=model; self.active_expert='system'
    def route_prompt(self,p):
        q=p.lower(); self.active_expert='math' if any(x in q for x in ('math','aime','equation','\\boxed')) else 'code' if any(x in q for x in ('python','class','git','patch','def ')) else 'lore' if any(x in q for x in ('history','currency','balehan','lore')) else 'system'; return self.active_expert
class MoEDualBufferManager:
    def __init__(self,model,settings): self.model=model; self.settings=settings; self.lock=threading.Lock(); self.adapters_buffer_a={}; self.adapters_buffer_b={}; self._init()
    def _init(self):
        if MLX_AVAILABLE and self.model is not None:
            try:
                self.model.freeze(); self.adapters_buffer_b=dict(mlx.utils.tree_flatten(self.model.trainable_parameters()))
            except Exception: pass
    def swap_buffers_atomic(self):
        if MLX_AVAILABLE and self.model is not None and self.adapters_buffer_b:
            with self.lock,METAL_STREAM_LOCK:
                self.model.update(mlx.utils.tree_unflatten(list(self.adapters_buffer_b.items()))); mx.eval(self.model.parameters())
class MoEParameterDistiller:
    def __init__(self,output_dir='eval_results'): self.output_dir=output_dir
    def distill_expert_cluster(self,experts,weights=None):
        if not experts:return {}
        weights=weights or {k:1/len(experts) for k in experts}; total=sum(weights.values()) or 1.; out={}
        for key in {k for d in experts.values() for k in d}:
            vals=[(weights.get(n,0)/total,d[key]) for n,d in experts.items() if key in d]
            if MLX_AVAILABLE and vals:
                acc=mx.zeros_like(vals[0][1]).astype(mx.float32)
                for w,v in vals: acc=acc+v.astype(mx.float32)*w
                mx.eval(acc); out[key]=acc
        return out
class DialogueTimelineGraphIngester:
    def __init__(self,kg): self.kg=kg
    def ingest_developer_sessions(self):
        rows=[('Ternary-Bonsai-27B','quantization_format','1.58-bit ternary MLX','ml_architecture'),('MLX Metal','uses','Apple unified memory','ml_architecture'),('TensorGraphDSL','supports','fold scale fuse','eval')]
        for s,p,t,x in rows:self.kg.insert_triple(s,p,t,1.0,x)
        return len(rows)
class GitWorktreeScratchpad:
    def __init__(self,base_repo_dir): self.base_repo_dir=base_repo_dir
    def create_worktree(self):
        d=tempfile.mkdtemp(prefix='git_worktree_'); subprocess.run(['git','worktree','add','--detach',d,'HEAD'],cwd=self.base_repo_dir,capture_output=True); return d
    def cleanup_worktree(self,d): subprocess.run(['git','worktree','remove','--force',d],cwd=self.base_repo_dir,capture_output=True); shutil.rmtree(d,ignore_errors=True)

class UnifiedMasterEngine:
    def __init__(self,settings:Optional[EngineSettings]=None):
        self.settings=settings or EngineSettings(); self.kg=RelationalKnowledgeGraph(self.settings.db_path); self.sandbox=POSIXHardenedSandbox(self.settings.sandbox_timeout_seconds,self.settings.sandbox_max_memory_mb); self.mcp=FastMCPDispatcher(self.sandbox,self.kg); self.lif=NeuromorphicLIFController(); self.drafter=ASTPrefixTrieDrafter(); self.h2o=H2OKVCacheArena(self.settings.h2o_sink_tokens,self.settings.h2o_heavy_tokens,self.settings.h2o_max_budget); self.ogp_projector=GramSchmidtOGPProjector(self.settings.ogp_ortho_tolerance); self.mcts=SymbolicMCTSSearchEngine(self.sandbox); self.model=self.tokenizer=self.moe_manager=self.moe_router=self.grpo_trainer=self.ogp_daemon=None; self._initialize_runtime()
    def _initialize_runtime(self):
        if not MLX_AVAILABLE:return
        try:
            self.model,self.tokenizer=load(self.settings.mlx_model_path)
            self.moe_manager=MoEDualBufferManager(self.model,self.settings); self.moe_router=HierarchicalMoERouter(self.model); self.grpo_trainer=GRPOTrainingEngine(self.model,self.tokenizer,self.sandbox)
            if self.settings.enable_awake_ogp_daemon:
                self.ogp_daemon=ProjectedSleepConsolidationDaemon(self.moe_manager,self.ogp_projector,self.kg,self.tokenizer,self.settings,METAL_STREAM_LOCK); self.ogp_daemon.start()
        except Exception:self.model=self.tokenizer=None
    def generate(self,prompt,max_tokens=128):
        if not MLX_AVAILABLE or self.model is None or self.tokenizer is None:return f'[Engine Offline Response for: {prompt[:40]}]'
        cache=inp=logits=step=None
        try:
            if hasattr(self.tokenizer,'apply_chat_template'):
                try: prompt=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
                except Exception: pass
            toks=self.tokenizer.encode(prompt); cache=make_prompt_cache(self.model); eos=getattr(self.tokenizer,'eos_token_id',None); eos=set(eos if isinstance(eos,(list,tuple,set)) else [eos] if eos is not None else []); out=[]
            with METAL_STREAM_LOCK:
                inp=mx.array([toks]); logits=self.model(inp,cache=cache); a=mx.argmax(logits[0,-1]); mx.eval(a); nxt=int(a.item())
                for _ in range(max_tokens):
                    if nxt in eos:break
                    out.append(nxt); step=self.model(mx.array([[nxt]]),cache=cache); a=mx.argmax(step[0,-1]); mx.eval(a); nxt=int(a.item())
            return self.tokenizer.decode(out)
        finally:
            cache=inp=logits=step=None
            if MLX_AVAILABLE:
                try:
                    if hasattr(mx,'clear_cache'):mx.clear_cache()
                    elif hasattr(mx,'metal') and hasattr(mx.metal,'clear_cache'):mx.metal.clear_cache()
                except Exception:pass