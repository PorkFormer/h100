"""Lazy process-local patches: never import CUDA libraries before Ray assigns a GPU."""
import importlib.abc,importlib.machinery,os,sys
class PatchLoader(importlib.abc.Loader):
    def __init__(self,wrapped,name):self.wrapped=wrapped;self.name=name
    def create_module(self,spec):return self.wrapped.create_module(spec) if hasattr(self.wrapped,'create_module') else None
    def exec_module(self,module):
        self.wrapped.exec_module(module)
        if os.environ.get('NCBR_AUDIT_DIR'):
            import json
            from pathlib import Path
            target=Path(os.environ['NCBR_AUDIT_DIR']);target.mkdir(parents=True,exist_ok=True)
            (target/f'{os.getpid()}_{self.name.replace(chr(46),chr(95))}_module.json').write_text(json.dumps(dict(module=self.name,source=module.__file__,pid=os.getpid(),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))
        if self.name=='vllm.platforms.interface':
            Platform=module.Platform;original=Platform.device_id_to_physical_device_id.__func__
            @classmethod
            def uuid_physical_id(cls,device_id):
                visible=os.environ.get(cls.device_control_env_var,'')
                if visible and visible.split(',')[device_id].startswith('GPU-'):
                    import pynvml
                    pynvml.nvmlInit()
                    try:return pynvml.nvmlDeviceGetIndex(pynvml.nvmlDeviceGetHandleByUUID(visible.split(',')[device_id]))
                    finally:pynvml.nvmlShutdown()
                return original(cls,device_id)
            Platform.device_id_to_physical_device_id=uuid_physical_id
        elif self.name=='verl.workers.engine_workers':
            import train_audit
class PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        enabled=(fullname=='vllm.platforms.interface' and os.environ.get('NCBR_UUID_COMPAT')=='1') or (fullname in ['verl.workers.engine_workers','verl.experimental.agent_loop.agent_loop','verl.experimental.reward_loop.reward_loop','verl.workers.rollout.vllm_rollout.vllm_async_server'] and os.environ.get('NCBR_AUDIT_DIR'))
        if enabled:
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec and spec.loader:spec.loader=PatchLoader(spec.loader,fullname)
            return spec
sys.meta_path.insert(0,PatchFinder())
