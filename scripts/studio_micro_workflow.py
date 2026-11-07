"""Canonical owned Studio micro workflow; telemetry is a disposable child.

Stage beside instance.json as workflow.py, or pass --run-root explicitly.
Historical run enclosures remain immutable; stage this version for future runs.
"""
import asyncio,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parent)
root=parser.parse_args().run_root.resolve()
meta=json.loads((root/'instance.json').read_text())
release=Path(meta['runtime_root'])
sys.path.insert(0,str(release/'src'))
os.environ.update(PYTHONPATH=str(release/'src'),SHELLSMITH_ROOT=str(release),PYTHONDONTWRITEBYTECODE='1')
from shellsmith.jobs.bundle import verify
from studio_compute.client import Client
from shellsmith.jobs.studio_monitor import OperationCollector,wait_for_training,best_effort_write,spawn_logged
from shellsmith.episode.gm_process.auth_identity import verify_guard,file_identity,registry_identity
from shellsmith.episode.gm_process.codex import CodexWorldConfig,CodexWorldProcess
from shellsmith.episode.gm_process.api_idle import api_idle_command
from shellsmith.episode.gm_process.python_runtime import current_python_runtime
verify(release,meta['source_sha256'])
started=time.time();clock=time.monotonic();children=[];service=None;trainer=None;monitor=None;client=None;active_phase=None;events=[]
meta.update(status='starting',owner_pid=os.getpid(),started=started,deadline_at=meta.get('absolute_deadline_at',started+meta['hard_deadline_seconds']))
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
 with temporary.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(temporary,path)
def record():
 if not best_effort_write(root/'instance.json',meta):
  meta['owner_status_write_degraded']=True
def phase(name,status):
 global active_phase
 value={'phase':name,'status':status,'timestamp':time.time()}
 if status=='started':value['started_at']=value['timestamp'];active_phase=name
 else:value['finished_at']=value['timestamp'];active_phase=None
 events.append(value)
 with (root/'phase-events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush()
 meta['phase']=active_phase;record()
def interrupted(*_):raise KeyboardInterrupt
signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
def owned():
 rows={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid,ppid,lstart,command'],text=True).splitlines()[1:]:
  fields=line.strip().split(None,2)
  if len(fields)==3:rows[int(fields[0])]=(int(fields[1]),fields[2])
 found={os.getpid()}
 while True:
  more={pid for pid,(ppid,_) in rows.items() if ppid in found}-found
  if not more:break
  found.update(more)
 return {pid:rows[pid][1] for pid in found-{os.getpid()}}
def spawn(command,log_name):
 p=spawn_logged(command,root/log_name,cwd=release,env=os.environ.copy())
 children.append(p);return p

def probe(resume=None):
 directory=root/'auth-probe';directory.mkdir(exist_ok=True,mode=0o700)
 import shutil
 from shellsmith.train.config import load_train_config
 gm=load_train_config(root/'train.toml').gm
 config=CodexWorldConfig(executable=Path(shutil.which('codex')),model=gm.model,reasoning_effort=gm.reasoning_effort,timeout_seconds=gm.timeout_seconds,api_idle_timeout_seconds=gm.api_idle_timeout_seconds,workspace_dir=directory,log_dir=directory,bootstrap_response_schema_path=directory/'unused-schema.json')
 command=CodexWorldProcess(config)._base_command(resume=resume is not None)+([resume,'-'] if resume else ['-'])
 name='resume' if resume else 'initial';begin=time.time()
 with api_idle_command(command,idle_seconds=gm.api_idle_timeout_seconds,turn_seconds=gm.timeout_seconds) as (api_command,transport):
  try:
   with (directory/(name+'.jsonl')).open('xb') as out,(directory/(name+'.stderr.log')).open('xb') as err:
    p=subprocess.Popen(api_command,cwd=directory,env=os.environ.copy(),stdin=subprocess.PIPE,stdout=out,stderr=err,process_group=0);children.append(p)
    try:p.communicate(b'Do not use tools. Reply with exactly the word OK.\n',timeout=gm.timeout_seconds)
    except BaseException:
     if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
     raise
  finally:
   write(directory/(name+'-api-transport.json'),{'idle_timeout_seconds':gm.api_idle_timeout_seconds,'turn_timeout_seconds':gm.timeout_seconds,'api_idle_timeouts':list(transport.timeouts)})
 rows=[json.loads(x) for x in (directory/(name+'.jsonl')).read_text().splitlines() if x.strip()]
 threads=[x['thread_id'] for x in rows if x.get('type')=='thread.started']
 messages=[x.get('item',{}).get('text','') for x in rows if x.get('type')=='item.completed' and x.get('item',{}).get('type')=='agent_message']
 tools=[x for x in rows if x.get('item',{}).get('type') in ('command_execution','file_change','mcp_tool_call','web_search')]
 receipt={'phase':name,'returncode':p.returncode,'seconds':time.time()-begin,'threads':threads,'response_ok':messages==['OK'],'tools_used':len(tools)}
 write(directory/(name+'-receipt.json'),receipt)
 if p.returncode or len(threads)!=1 or (resume and threads[0]!=resume) or messages!=['OK'] or tools:raise RuntimeError('Initial/resumed GM authentication probe failed; inspect preserved evidence.')
 return threads[0]

collector=OperationCollector(root)
def collect_operations():
 return collector.collect()
def monitor_failed(returncode):
 meta['monitoring']={'status':'degraded','returncode':returncode,'timestamp':time.time(),'training_continues':True}
 record()

def control_tick():
 global initial_recorded,initial
 # This is a required compute identity check, not optional telemetry.
 if not initial_recorded:
  try:client.client_info()
  except ValueError:return
  initial=client.call('inspect',{'hash_state':True,'hash_base':True},request_id=meta['run_id']+':initial-inspect')
  assert initial['optimizer_steps']==0 and initial['frozen_base_hash']==meta['expected_base_hash']
  best_effort_write(root/'compute-initial.json',initial);initial_recorded=True

def stop_service():
 global service
 if service and service.poll() is None:
  service.send_signal(signal.SIGTERM)
  try:service.wait(timeout=15)
  except subprocess.TimeoutExpired:service.kill();service.wait()

def start_service():
 global service,client,cap
 number=meta.get('service_starts',0);meta['service_starts']=number+1
 service=spawn([meta['python'],'-u','-m','studio_compute.cli','serve','--config',str(root/'service.toml')],f'service-{number}.log')
 meta['service_pid']=service.pid;record()
 client=Client(meta['endpoint'],meta['run_id'],timeout=180)
 until=time.monotonic()+180
 while True:
  if service.poll() is not None:raise RuntimeError('Owned compute exited during startup')
  try:cap=client.capabilities();break
  except OSError:
   if time.monotonic()>until:raise TimeoutError('Compute startup deadline')
   time.sleep(.25)
 assert cap['identity']['numerics']==meta['expected_numerics'] and cap['context_window']==262144
 assert cap['identity']['runtime_source_sha256']==meta['expected_runtime_sources']
 assert cap['runtime']['prefix_cache_mode']=='available_memory';write(root/'capabilities.json',cap)
 return cap

record()
try:
 phase('authentication_preflight','started')
 import shutil
 executable=shutil.which('codex');flags=verify_guard(executable);assert 'model_provider="shellsmith_codex_oauth"' in flags
 guard=json.loads(Path(os.environ['SHELLSMITH_GM_AUTH_GUARD']).read_text());identity=(registry_identity(os.environ['CODEX_HOME'],guard['source']) if guard.get('credential_store')=='bladerunner_registry' else file_identity(os.environ['CODEX_HOME']))
 assert identity['account_fingerprint']==guard['account_fingerprint']
 write(root/'auth-selected.json',identity)
 runtime=current_python_runtime();runtime.probe();write(root/'gm-python-runtime.json',{'executable':str(runtime.executable),'prefix':str(runtime.prefix)})
 import tomllib
 config=tomllib.loads((root/'train.toml').read_text())
 from shellsmith.train.config import load_train_config
 load_train_config(root/'train.toml')
 sdk=Path(config['player']['pi_sdk_module_path']);assert sdk.is_file()
 probe(probe());phase('authentication_preflight','finished')
 phase('service_startup','started');start_service();phase('service_startup','finished')
 phase('native_compute_preflight','started')
 from native_preflight import run as native_preflight
 initial=native_preflight(root,meta,start_service,stop_service,write)
 from shellsmith.train.config import load_train_config
 from shellsmith.mlx.generation import reasoning_budget,completion_budget
 parsed=load_train_config(root/'train.toml')
 assert parsed.gm.reasoning_effort=='medium'
 assert reasoning_budget(parsed.studio)==4096 and completion_budget(parsed.studio)==36867
 assert parsed.studio.context_window==262144 and parsed.gm.max_world_steps==256
 assert parsed.adaptive.initial_attempts==parsed.adaptive.maximum_attempts==parsed.adaptive.parallelism==4
 write(root/'protocol-preflight.json',{'served_tokenizer_hash':cap['tokenizer_hash'],'reasoning_effort':parsed.studio.reasoning_effort,'reasoning_tokens':4096,'response_tokens':32768,'completion_reserve':36867,'context_window':262144,'max_world_steps':256,'attempts':4,'training_updates':1,'native_two_pass_tool_sample_passed':True,'native_tiny_update_and_fresh_process_recovery_passed':True,'timestamp':time.time()})
 phase('native_compute_preflight','finished');phase('rollout_and_training','started')
 client=Client(meta['endpoint'],meta['run_id'],timeout=180)
 initial_recorded=False
 trainer=spawn([meta['python'],'-u','-m','shellsmith','train','--config',str(root/'train.toml'),'--log-dir',str(root/'train')],'train-console.log')
 meta.update(trainer_pid=trainer.pid,status='running',ready=time.time());record()
 try:
  monitor=spawn([meta['python'],'-u','-m','shellsmith.jobs.studio_monitor','--run-root',str(root),'--service-pid',str(service.pid),'--initial-swap',str(initial['metrics']['swap_used_bytes'])],'monitor-console.log')
  meta['monitor_pid']=monitor.pid;meta['monitoring']={'status':'starting'};record()
 except Exception as error:
  meta['monitor_start_error']=type(error).__name__+': '+str(error)
  monitor_failed(None)
 wait_for_training(trainer,service,monitor,on_tick=control_tick,on_monitor_failure=monitor_failed)
 if monitor and monitor.poll() is None:
  monitor.terminate()
  try:monitor.wait(timeout=3)
  except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
 meta.update(trainer_returncode=trainer.returncode,trainer_finished=time.time());record()
 if active_phase:phase(active_phase,'finished')
 phase('final_verification','started')
 operations=collector.collect(persist=False);state_path=root/'train/trainer_state.json'
 state=json.loads(state_path.read_text()) if state_path.exists() else {}
 if trainer.returncode==0 and state.get('step')==1:
  from shellsmith.train.studio import load_renderer
  renderer=load_renderer(client,cap)
  rendered=renderer.render({'messages':[{'role':'user','content':'Reply with READY.'}],'tools':[]},thinking=False)
  ids=rendered[0].to_ints()
  post=client.call('sample',{'policy':state['active_policy_ref'],'prompt_token_ids':ids,'max_tokens':8,'cache_precision':'bf16','seed':17},request_id=meta['run_id']+':published-sample')
  assert post['policy_revision']==1;write(root/'published-sample.json',post)
 final=client.call('inspect',{'hash_state':True,'hash_base':True},request_id=meta['run_id']+':final-inspect');write(root/'compute-final.json',final)
 assert final['frozen_base_hash']==initial['frozen_base_hash']
 operations=collector.collect(persist=False)
 if trainer.returncode==0 and state.get('step')==1:
  optimizer=[x for x in operations if x['kind']=='optim_step' and x['status']=='succeeded' and x['client']==meta['run_id']];backward=[x for x in operations if x['kind']=='forward_backward' and x['status']=='succeeded' and x['client']==meta['run_id']]
  assert len(optimizer)==len(backward)==1 and final['optimizer_steps']==1
  assert math.isfinite(optimizer[0]['result']['optim_metrics']['gradient_norm']) and math.isfinite(backward[0]['result']['forward_metrics']['loss'])
  assert any(x['kind']=='publish' and x['client']==meta['run_id'] and x['status']=='succeeded' and x['result']['revision']==1 and x['result']['policy']==state['active_policy_ref'] for x in operations)
  meta['workflow_status']='passed'
 else:meta['workflow_status']='incomplete_no_update' if trainer.returncode==0 else 'committed_update_then_workflow_failure' if state.get('step')==1 else 'failed_infrastructure'
 meta.update(status='finished' if trainer.returncode==0 else 'failed',finished=time.time());phase('final_verification','finished')
except KeyboardInterrupt:
 meta.update(status='interrupted',stop_reason='owner_deadline_or_signal',finished=time.time())
except BaseException as e:
 meta.update(status='failed',error=type(e).__name__+': '+str(e),finished=time.time())
 import traceback
 best_effort_write(root/'owner-error.json',{'traceback':traceback.format_exc()})
finally:
 try:snapshot=owned()
 except Exception:snapshot={}
 meta['owned_child_identities']=snapshot
 for process in (monitor,trainer,service):
  if process and process.poll() is None:
   try:os.killpg(process.pid,signal.SIGINT if process is trainer else signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=8)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()
 for process in children:
  if process.poll() is None:
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:pass
 live={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid,ppid,lstart,command'],text=True).splitlines()[1:]:
  fields=line.strip().split(None,2)
  if len(fields)==3:live[int(fields[0])]=fields[2]
 cleaned=[]
 for pid,identity in snapshot.items():
  if live.get(pid)==identity:
   try:os.kill(pid,signal.SIGKILL);cleaned.append(pid)
   except ProcessLookupError:pass
 try:collect_operations()
 except Exception:pass
 meta.update(elapsed_seconds=time.monotonic()-clock,cleanup_fallback_pids=cleaned,service_returncode=service.returncode if service else None,finished=meta.get('finished',time.time()))
 record();best_effort_write(root/'cleanup.json',{'owned_snapshot':snapshot,'fallback_killed':cleaned,'service_returncode':meta['service_returncode'],'trainer_returncode':meta.get('trainer_returncode'),'finished':time.time()})
 print(json.dumps({k:meta.get(k) for k in ('status','trainer_returncode','elapsed_seconds','error','stop_reason')}),flush=True)
raise SystemExit(0 if meta['status']=='finished' else 1)
