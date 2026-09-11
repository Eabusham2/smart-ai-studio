"""Procedural symbolic task synthesis plus verifier-backed Monte Carlo Tree Search."""
from __future__ import annotations
import math,random
from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional,Sequence,Tuple

@dataclass
class DiscoveryTask:
    task_id:str;domain:str;prompt:str;expected:str;candidates:List[str];metadata:Dict[str,Any]=field(default_factory=dict)
    @property
    def expression(self):return str(self.metadata.get("expression",self.prompt))
@dataclass
class MCTSNode:
    candidate:Optional[str];parent:Optional["MCTSNode"]=None;children:List["MCTSNode"]=field(default_factory=list);visits:int=0;total_reward:float=0.0
    @property
    def mean_reward(self):return self.total_reward/max(1,self.visits)
class DeterministicSymbolicEnvironment:
    @staticmethod
    def _normalize(x):return "".join(str(x).lower().split())
    def reward(self,task,candidate):return 1.0 if self._normalize(candidate)==self._normalize(task.expected) else 0.0
    @staticmethod
    def rotate_scale(values,k,scale):
        v=list(map(int,values));k%=len(v) if v else 1;return [x*int(scale) for x in (v[k:]+v[:k])]
    @staticmethod
    def matrix_multiply(a,b):return [[sum(int(a[i][k])*int(b[k][j]) for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    @staticmethod
    def compose_permutations(p,q):return [int(p[int(q[i])]) for i in range(len(q))]
class ProceduralTaskGenerator:
    def __init__(self,seed=7331):self.rng=random.Random(seed);self.env=DeterministicSymbolicEnvironment();self.counter=0
    def _distractors(self,expected,count=5):
        c={str(expected)}
        if isinstance(expected,list):
            c.add(str(list(reversed(expected))))
            if expected:c.add(str(expected[1:]+expected[:1]))
        while len(c)<count:c.add(str(self.rng.randint(-9,99)))
        c=list(c);self.rng.shuffle(c);return c
    def _dsl_task(self):
        n=self.rng.randint(3,6);vals=[self.rng.randint(-5,9) for _ in range(n)];k=self.rng.randint(1,n-1);scale=self.rng.choice([2,3,4,-1]);exp=self.env.rotate_scale(vals,k,scale);expr=f"{vals} >>~fold({k}) <#>scale({scale})"
        return DiscoveryTask(f"proc-dsl-{self.counter}","TensorGraphDSL",f"Using fold(k)=rotate left and scale(s)=multiply each element, evaluate exactly: {expr}",str(exp),self._distractors(exp),{"expression":expr})
    def _matrix_task(self):
        a=[[self.rng.randint(-3,4) for _ in range(2)] for _ in range(2)];b=[[self.rng.randint(-3,4) for _ in range(2)] for _ in range(2)];exp=self.env.matrix_multiply(a,b)
        return DiscoveryTask(f"proc-matrix-{self.counter}","MatrixGraph",f"Compute exact 2x2 A@B for A={a}, B={b}.",str(exp),self._distractors(exp),{"a":a,"b":b})
    def _nonabelian_task(self):
        n=self.rng.choice([3,4,5]);p=list(range(n));q=list(range(n));self.rng.shuffle(p);self.rng.shuffle(q);exp=str(self.env.compose_permutations(p,q));c={exp,str(self.env.compose_permutations(q,p)),str(p),str(q)}
        while len(c)<5:
            d=list(range(n));self.rng.shuffle(d);c.add(str(d))
        c=list(c);self.rng.shuffle(c);return DiscoveryTask(f"proc-nonabelian-{self.counter}","NonAbelianAlgebra",f"Using zero-based image notation and (p o q)(i)=p[q[i]], compute p o q for p={p}, q={q}.",exp,c,{"p":p,"q":q})
    def generate(self,count=24):
        out=[];builders=(self._dsl_task,self._matrix_task,self._nonabelian_task)
        for i in range(max(1,int(count))):self.counter+=1;out.append(builders[i%3]())
        return out
class ProceduralMCTSDiscovery:
    def __init__(self,sandbox=None,knowledge_graph=None,simulations=32,exploration=math.sqrt(2),seed=7331):self.sandbox=sandbox;self.knowledge_graph=knowledge_graph;self.simulations=max(4,int(simulations));self.exploration=float(exploration);self.generator=ProceduralTaskGenerator(seed);self.environment=DeterministicSymbolicEnvironment()
    def _ucb(self,n,parent_visits):return float("inf") if n.visits==0 else n.mean_reward+self.exploration*math.sqrt(math.log(max(2,parent_visits))/n.visits)
    def search_task(self,task):
        root=MCTSNode(None);root.children=[MCTSNode(c,root) for c in task.candidates]
        for _ in range(self.simulations):
            child=max(root.children,key=lambda n:self._ucb(n,root.visits+1));r=self.environment.reward(task,child.candidate or "");child.visits+=1;child.total_reward+=r;root.visits+=1;root.total_reward+=r
        w=max(root.children,key=lambda n:(n.mean_reward,n.visits));return str(w.candidate),float(w.mean_reward),int(root.visits)
    def discover_batch(self,count=24):
        out=[]
        for task in self.generator.generate(count):
            ans,reward,visits=self.search_task(task);rec={"task_id":task.task_id,"domain":task.domain,"prompt":task.prompt,"expected":task.expected,"discovered":ans,"reward":reward,"visits":visits,"metadata":task.metadata};out.append(rec)
            if reward>=1 and self.knowledge_graph is not None:
                try:self.knowledge_graph.insert_triple(task.task_id,"verified_invariant",ans,weight=reward,session_id="procedural_mcts")
                except Exception:pass
        return out
    def search_best_invariant(self,expression):
        if self.sandbox is not None:
            try:
                result=self.sandbox.evaluate_dsl_expression(expression)
                if result is not None:return str(result),1.0,1
            except Exception:pass
        task=DiscoveryTask("compat-invariant","TensorGraphDSL",expression,expression,[expression]);return self.search_task(task)
