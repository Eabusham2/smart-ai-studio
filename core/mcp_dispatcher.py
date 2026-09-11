"""Small in-process JSON-RPC 2.0/MCP-style dispatcher for deterministic local tools."""
from __future__ import annotations
import json
from typing import Any,Callable,Dict
class InProcessMCPDispatcher:
    def __init__(self,sandbox,knowledge_graph=None):
        self.sandbox=sandbox;self.knowledge_graph=knowledge_graph;self.tools={}
        self.register_tool("dsl_evaluate",self._dsl,{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]})
        self.register_tool("python_verify",self._python_verify,{"type":"object","properties":{"code":{"type":"string"},"tests":{"type":"string"}},"required":["code","tests"]})
        self.register_tool("matrix_vector_dot",self._matrix_vector_dot,{"type":"object","properties":{"vector_a":{"type":"array"},"vector_b":{"type":"array"}},"required":["vector_a","vector_b"]})
    def register_tool(self,name:str,handler:Callable[[Dict[str,Any]],Any],schema:Dict[str,Any]):self.tools[name]={"handler":handler,"inputSchema":schema}
    def _dsl(self,args):return self.sandbox.evaluate_dsl_expression(str(args.get("expression","")))
    def _python_verify(self,args):
        r=self.sandbox.execute_python_code(str(args.get("code","")),str(args.get("tests","")));return {"passed":r.passed,"stdout":r.output,"error":r.error,"execution_time_ms":r.execution_time_ms}
    @staticmethod
    def _matrix_vector_dot(args):
        a=list(args.get("vector_a",[]));b=list(args.get("vector_b",[]))
        if len(a)!=len(b):raise ValueError("vectors must have equal length")
        return sum(float(x)*float(y) for x,y in zip(a,b))
    def _resource(self,uri):
        if uri=="memory://graph/stats" and self.knowledge_graph is not None:return self.knowledge_graph.stats()
        if uri.startswith("memory://graph/recall?") and self.knowledge_graph is not None:return self.knowledge_graph.recall(uri.split("?",1)[1])
        raise KeyError(uri)
    @staticmethod
    def _validate(schema,args):
        if not isinstance(args,dict):raise TypeError("arguments must be an object")
        for key in schema.get("required",[]):
            if key not in args:raise ValueError(f"missing required argument: {key}")
    def dispatch(self,request):
        rid=request.get("id")
        if request.get("jsonrpc")!="2.0":return {"jsonrpc":"2.0","id":rid,"error":{"code":-32600,"message":"Invalid Request"}}
        try:
            method=request.get("method")
            if method=="tools/list":return {"jsonrpc":"2.0","id":rid,"result":{"tools":[{"name":n,"inputSchema":v["inputSchema"]} for n,v in sorted(self.tools.items())]}}
            if method=="tools/call":
                p=request.get("params") or {};name=p.get("name");args=p.get("arguments") or {}
                if name not in self.tools:raise KeyError(f"unknown tool: {name}")
                spec=self.tools[name];self._validate(spec["inputSchema"],args);value=spec["handler"](args);return {"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":json.dumps(value,default=str)}],"structuredContent":value}}
            if method=="resources/read":
                uri=(request.get("params") or {}).get("uri","");value=self._resource(uri);return {"jsonrpc":"2.0","id":rid,"result":{"contents":[{"uri":uri,"mimeType":"application/json","text":json.dumps(value,default=str)}]}}
            return {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"Method not found"}}
        except KeyError as exc:return {"jsonrpc":"2.0","id":rid,"error":{"code":-32602,"message":str(exc)}}
        except Exception as exc:return {"jsonrpc":"2.0","id":rid,"error":{"code":-32000,"message":f"{type(exc).__name__}: {exc}"}}
    def handle_json_rpc(self,request_json):
        try:req=json.loads(request_json)
        except Exception as exc:return json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":str(exc)}})
        return json.dumps(self.dispatch(req))
