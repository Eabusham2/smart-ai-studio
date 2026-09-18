"""Small flow simulations; no scores claim a real model inference pass."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def env(tmp_path,monkeypatch):
    # Load this isolated addition without booting the project's heavyweight models.
    core = types.ModuleType("core"); core.__path__=[str(ROOT/"core")]
    config = types.ModuleType("config"); config.__path__=[]
    paths = types.ModuleType("config.paths");paths.get_portable_data_dir=lambda:str(tmp_path/"state")
    for name,mod in [("core",core),("config",config),("config.paths",paths)]:monkeypatch.setitem(sys.modules,name,mod)
    modules={}
    for name in ["media_learning","media_review","media_orchestrator"]:
        spec=importlib.util.spec_from_file_location("core."+name,ROOT/"core"/(name+".py"))
        mod=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,"core."+name,mod);spec.loader.exec_module(mod);modules[name]=mod
    class Text:
        def __init__(self):
            self.is_model_loaded=True;self.active_model_name="existing-text";self.active_backend="gguf"
            self.gguf_backend=types.SimpleNamespace(model_path="same-model.gguf")
            self.lora_adapter_path="same-adapter";self.unloads=0;self.loads=[];self.fail_restore=False;self.outputs=["ordinary reply"]
        def unload_model(self):self.unloads+=1;self.is_model_loaded=False
        def load_model(self,name,model_path=None,backend=None):
            self.loads.append((name,model_path,backend,self.lora_adapter_path))
            self.is_model_loaded=not self.fail_restore
            return {"status":"error" if self.fail_restore else "loaded"}
        def stream_solve(self,prompt,history=None,cancel_event=None):yield self.outputs.pop(0) if self.outputs else "done"
        def solve(self,*args,**kwargs):return "refined prompt",{}
    class Media:
        def __init__(self):self.calls=[];self.fail=False;self.empty=False
        def unload_model(self):pass
        def generate_image(self,info,prompt,output_path=None,**kwargs):
            self.calls.append(info["repo_id"])
            if self.fail:raise RuntimeError("simulated OOM")
            Path(output_path).write_bytes(b"test-artifact" if not self.empty else b"")
            return {"status":"success","path":output_path}
        def generate_video(self,*a,**k):return self.generate_image(*a,**k)
    class Tools:
        def list_available_tools(self):return [{"name":"existing_tool"}]
        def execute_tool(self,name,args):return True,"untouched"
    app=types.SimpleNamespace(engine=Text(),media_engine=Media(),audio_engine=Media(),tools=Tools(),root=types.SimpleNamespace(after=lambda delay,fn:fn()),workspace_dir=str(tmp_path),active_tab_id="text",is_model_loaded=True,cancel_event=threading.Event(),_memory_limit_enabled=False,_sync_memory_watchdog=lambda:None,_append_ai_message=lambda message:None)
    app.models_config={"text":{"model_type":"text"},"image":{"model_type":"image","repo_id":"tested-image","real_media_engine":True,"vram":"4 GB"},"audio":{"model_type":"audio","repo_id":"tested-audio"},"video":{"model_type":"video","repo_id":"tested-video"}}
    controller=modules['media_orchestrator'].MediaController(app)
    controller._room=lambda needed:True
    return controller,app,modules,tmp_path

def test_existing_tools_remain_intact(env):
    c,a,_,_=env
    assert a.tools.execute_tool("existing_tool",{})==(True,"untouched")
    assert "media_generate" in [r["name"] for r in a.tools.list_available_tools()]

def test_fits_keeps_text_resident(env):
    c,a,_,_=env;r=c.call("media_generate",{"kind":"image","prompt":"test"})
    assert r["status"]=="success" and Path(r["path"]).is_file()
    assert a.engine.unloads==0 and not c.busy and not r['weights_updated']

def test_pause_resumes_exact_text_identity(env):
    c,a,_,_=env;c._room=lambda needed:not a.engine.is_model_loaded
    r=c.call("media_generate",{"kind":"image","prompt":"test"})
    assert r["status"]=="success"
    assert a.engine.unloads==1
    assert a.engine.loads==[("existing-text","same-model.gguf","gguf","same-adapter")]

def test_media_failure_still_restores_text(env):
    c,a,_,_=env;c._room=lambda needed:not a.engine.is_model_loaded;a.media_engine.fail=True
    r=c.call("media_generate",{"kind":"image","prompt":"test"})
    assert r["status"]=="error" and a.engine.is_model_loaded and not c.busy

def test_insufficient_room_aborts_after_restoring(env):
    c,a,_,_=env;c._room=lambda needed:False
    r=c.call("media_generate",{"kind":"image","prompt":"test"})
    assert r["status"]=="error" and a.engine.is_model_loaded and not a.media_engine.calls

def test_restore_failure_is_not_hidden(env):
    c,a,_,_=env;c._room=lambda needed:not a.engine.is_model_loaded;a.engine.fail_restore=True
    r=c.call("media_generate",{"kind":"image","prompt":"test"})
    assert r["status"]=="error" and not a.is_model_loaded and "restored" in r["error"]

def test_no_artifact_no_success(env):
    c,a,_,_=env;a.media_engine.empty=True
    assert c.call("media_generate",{"kind":"image","prompt":"test"})['status']=='error'

def test_cancel_does_not_load_models(env):
    c,a,_,_=env;a.cancel_event.set()
    assert c.call("media_generate",{"kind":"image","prompt":"test"})['status']=='error'
    assert not a.media_engine.calls and not a.engine.loads

def test_imported_path_cannot_be_reviewed_without_artifact(env):
    c,_,_,_=env
    assert c.call("media_review",{"artifact_id":"/etc/passwd"})['status']=='error'

def test_normal_chat_and_original_history_preserved(env):
    c,a,_,_=env;hist=[{"role":"user","content":"old turn"}]
    assert ''.join(c.stream_solve("hello",history=hist))=="ordinary reply"
    assert hist==[{"role":"user","content":"old turn"}] and not c.busy

def test_model_can_call_existing_media_engine(env):
    c,a,_,_=env
    a.engine.outputs=['<media_call>{"name":"media_generate","arguments":{"kind":"image","prompt":"test"}}</media_call>','Here is the result.']
    text=''.join(c.stream_solve("make a picture"))
    assert a.media_engine.calls==['tested-image'] and 'artifact_id' in text and text.endswith('Here is the result.')

def test_tool_loop_is_bounded(env):
    c,a,_,_=env;a.engine.outputs=['<media_call>{"name":"media_list_models","arguments":{}}</media_call>']*10
    assert 'limit reached' in ''.join(c.stream_solve('test'))
    assert len(a.engine.outputs)==7 and not c.busy

def test_model_cannot_silently_train(env):
    c,a,_,_=env
    r=c.call('media_learn',{'model_id':'image','dataset':'a.jsonl'})
    assert r['status']=='error' and 'explicit' in r['error'] and not r['weights_updated']

def test_unmatched_media_backend_is_not_reported_trained(env):
    c,a,_,_=env
    # The fixture uses synthetic repo IDs with no registered architecture trainer.
    # Real audio/video families may train when a matching backend is installed.
    for mid in ['audio','video']:
        r=c.call('media_learn',{'model_id':mid,'dataset':'a.jsonl'},allow_update=True)
        assert r['status']=='unsupported' and not r['weights_updated']

def test_text_learn_untouched(env):
    c,_,_,_=env;assert c.handle_command('/learn algebra') is None

def test_dataset_requires_real_captioned_workspace_media(env):
    c,_,_,tmp=env
    (tmp/'a.bin').write_bytes(b'actual media fixture')
    (tmp/'data.jsonl').write_text(json.dumps({'path':'a.bin','caption':'caption'})+'\n')
    assert len(c.learning.read_samples('data.jsonl',tmp))==1
    with pytest.raises(ValueError):c.learning.read_samples('../escape.jsonl',tmp)

def test_no_perception_is_not_a_grade(env):
    c,a,mods,tmp=env;p=tmp/'bad.png';p.write_bytes(b'not an image')
    r=mods['media_review'].review_artifact({'path':str(p),'kind':'image','prompt':'cat'})
    assert not r['perception_available'] and 'alignment_cosine' not in r


def test_pause_rejected_during_text_consolidation(env):
    c,a,_,_=env;c._room=lambda needed:not a.engine.is_model_loaded
    a.engine.awake_consolidator=types.SimpleNamespace(lock=threading.Lock(),is_consolidating=True)
    r=c.call('media_generate',{'kind':'image','prompt':'test'})
    assert r['status']=='error' and not a.engine.unloads and not c.busy

def test_training_mutex_released_after_media_failure(env):
    c,a,_,_=env;c._room=lambda needed:not a.engine.is_model_loaded
    lock=threading.Lock();a.engine.awake_consolidator=types.SimpleNamespace(lock=lock,is_consolidating=False)
    a.media_engine.fail=True
    r=c.call('media_generate',{'kind':'image','prompt':'test'})
    assert r['status']=='error' and lock.acquire(blocking=False)
    lock.release()

def test_no_text_continuation_after_failed_restore(env):
    c,a,_,_=env;c._room=lambda needed:not a.engine.is_model_loaded;a.engine.fail_restore=True
    a.engine.outputs=['<media_call>{"name":"media_generate","arguments":{"kind":"image","prompt":"test"}}</media_call>','must not run']
    text=''.join(c.stream_solve('make an image'))
    assert 'continuation stopped' in text and a.engine.outputs==['must not run']
