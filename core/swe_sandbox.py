"""Hardened multi-file verification sandbox used by benchmark/SWE tasks."""
from __future__ import annotations
import os,platform,re,shutil,subprocess,sys,tempfile,time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict,Optional
try:
    import resource
    HAS_RESOURCE=True
except ImportError:
    resource=None;HAS_RESOURCE=False
@dataclass
class SandboxResult:
    passed:bool;execution_time_ms:float;output:str;error:Optional[str]=None;reward:float=0.0
class HardenedSWESandbox:
    def __init__(self,timeout_sec=4.0,max_memory_mb=512,max_files=64):self.timeout=float(timeout_sec);self.max_memory_mb=max(512,int(max_memory_mb));self.max_files=int(max_files)
    def _preexec(self):
        if not HAS_RESOURCE or platform.system()=="Windows":return
        mem=self.max_memory_mb*1024*1024
        try:
            if hasattr(resource,"RLIMIT_AS") and platform.system()!="Darwin":resource.setrlimit(resource.RLIMIT_AS,(mem,mem))
            if hasattr(resource,"RLIMIT_DATA"):resource.setrlimit(resource.RLIMIT_DATA,(mem,mem))
            if hasattr(resource,"RLIMIT_CPU"):
                cpu=max(1,int(self.timeout)+1);resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
            if hasattr(resource,"RLIMIT_NOFILE"):resource.setrlimit(resource.RLIMIT_NOFILE,(self.max_files,self.max_files))
            if hasattr(resource,"RLIMIT_CORE"):resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        except Exception:pass
    @staticmethod
    def _network_guard(directory):
        Path(directory,"sitecustomize.py").write_text("import os\nif os.getenv('SMARTAI_DISABLE_NETWORK')=='1':\n import socket\n class _BlockedSocket(socket.socket):\n  def connect(self,*a,**k): raise PermissionError('network disabled in verifier')\n  def connect_ex(self,*a,**k): raise PermissionError('network disabled in verifier')\n socket.socket=_BlockedSocket\n",encoding="utf-8")
    def _run(self,cmd,cwd,shell=False):
        start=time.perf_counter();env=dict(os.environ);env["SMARTAI_DISABLE_NETWORK"]="1";env["PYTHONNOUSERSITE"]="1";env["PYTHONPATH"]=cwd+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            cp=subprocess.run(cmd,cwd=cwd,env=env,shell=shell,capture_output=True,text=True,timeout=self.timeout,preexec_fn=self._preexec if HAS_RESOURCE and platform.system()!="Windows" else None);dt=(time.perf_counter()-start)*1000;return SandboxResult(cp.returncode==0,dt,cp.stdout,None if cp.returncode==0 else cp.stderr[-2000:],1.0 if cp.returncode==0 else 0.0)
        except subprocess.TimeoutExpired as exc:return SandboxResult(False,(time.perf_counter()-start)*1000,exc.stdout or "","ProcessTimeout",0.0)
        except Exception as exc:return SandboxResult(False,(time.perf_counter()-start)*1000,"",f"{type(exc).__name__}: {exc}",0.0)
    def execute_python_code(self,code,tests):
        root=tempfile.mkdtemp(prefix="smartai_py_jail_")
        try:
            self._network_guard(root);Path(root,"case.py").write_text("import math,json,collections,itertools\nimport socket as _socket\nclass _BlockedSocket(_socket.socket):\n    def connect(self,*a,**k): raise PermissionError('network disabled in verifier')\n    def connect_ex(self,*a,**k): raise PermissionError('network disabled in verifier')\n_socket.socket=_BlockedSocket\n"+code+"\n"+tests+"\n",encoding="utf-8");return self._run([sys.executable,"-I","case.py"],root)
        finally:shutil.rmtree(root,ignore_errors=True)
    def verify_git_diff_patch(self,repo_structure:Dict[str,str],patch_text:str,test_cmd:str):
        root=tempfile.mkdtemp(prefix="smartai_swe_jail_");start=time.perf_counter()
        try:
            if len(repo_structure)>self.max_files:return SandboxResult(False,0,"","too many files",0)
            self._network_guard(root)
            for rel,body in repo_structure.items():
                rp=Path(rel)
                if rp.is_absolute() or ".." in rp.parts:return SandboxResult(False,0,"","unsafe path",0)
                dest=Path(root)/rp;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(str(body),encoding="utf-8")
            blocks=re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```",patch_text,re.I);patch=(blocks[-1] if blocks else patch_text).strip()+"\n";Path(root,"task.patch").write_text(patch,encoding="utf-8")
            subprocess.run(["git","init","-q"],cwd=root,capture_output=True,timeout=2);ap=subprocess.run(["git","apply","--check","task.patch"],cwd=root,capture_output=True,text=True,timeout=2)
            if ap.returncode:return SandboxResult(False,(time.perf_counter()-start)*1000,ap.stdout,ap.stderr[-2000:],0)
            ap=subprocess.run(["git","apply","task.patch"],cwd=root,capture_output=True,text=True,timeout=2)
            if ap.returncode:return SandboxResult(False,(time.perf_counter()-start)*1000,ap.stdout,ap.stderr[-2000:],0)
            return self._run(test_cmd,root,shell=True)
        except Exception as exc:return SandboxResult(False,(time.perf_counter()-start)*1000,"",f"{type(exc).__name__}: {exc}",0)
        finally:shutil.rmtree(root,ignore_errors=True)
    @staticmethod
    def evaluate_dsl_expression(expr):
        m=re.match(r"\s*\[([^\]]*)\]",expr)
        if not m:return None
        try:arr=[int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        except Exception:return None
        rest=expr[m.end():]
        while rest.strip():
            rest=rest.lstrip();m=re.match(r">>~fold\((\d+)\)",rest)
            if m:
                if arr:k=int(m.group(1))%len(arr);arr=arr[k:]+arr[:k]
                rest=rest[m.end():];continue
            m=re.match(r"<#>scale\((-?\d+)\)",rest)
            if m:
                s=int(m.group(1));arr=[x*s for x in arr];rest=rest[m.end():];continue
            m=re.match(r"@fuse\s*\[([^\]]*)\]",rest)
            if m:
                try:rhs=[int(x.strip()) for x in m.group(1).split(",") if x.strip()]
                except Exception:return None
                if len(rhs)!=len(arr):return None
                arr=[a+b for a,b in zip(arr,rhs)];rest=rest[m.end():];continue
            return None
        return arr
    def create_local_worktree(self,repo_dir):
        dest=tempfile.mkdtemp(prefix="smartai_worktree_");shutil.rmtree(dest)
        try:
            cp=subprocess.run(["git","worktree","add","--detach",dest,"HEAD"],cwd=repo_dir,capture_output=True,text=True,timeout=8);return dest if cp.returncode==0 else None
        except Exception:return None
    def cleanup_local_worktree(self,repo_dir,path):
        if path:subprocess.run(["git","worktree","remove","--force",path],cwd=repo_dir,capture_output=True);shutil.rmtree(path,ignore_errors=True)
