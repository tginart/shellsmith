"""Canonical deadline supervisor; terminal receipts override stale workflow state."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parent)
root=parser.parse_args().run_root.resolve()
meta=json.loads((root/'instance.json').read_text())
if meta.get('runtime_root'):sys.path.insert(0,str(Path(meta['runtime_root'])/'src'))
from shellsmith.jobs.studio_monitor import effective_run_state,best_effort_write
from shellsmith.jobs.storage import process_identity,alive
if (root/'launch.json').exists() or (root/'owner-limit.json').exists():
 raise RuntimeError('One-shot owner: existing launch is never restarted.')
def write(name,value):
 target=root/name;tmp=target.with_suffix(target.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,target)
def rows():
 result={}
 for line in subprocess.check_output(['/bin/ps','-axo','pid,ppid,lstart,command'],text=True).splitlines()[1:]:
  fields=line.strip().split(None,2)
  if len(fields)==3:result[int(fields[0])]=(int(fields[1]),fields[2])
 return result
started=time.time();clock=time.monotonic();remaining=meta['absolute_deadline_at']-started
if remaining<=0:raise RuntimeError('No time remains for the authorized run; deadline is never reset.')
end=clock+remaining;grace=min(20,remaining/4);known={};reason=None;signalled=False;forced=[];tracking_errors=0
owner=None
def track():
 global tracking_errors
 if owner is None:return {}
 try:current=rows()
 except (OSError,subprocess.SubprocessError):
  tracking_errors+=1
  return {}
 # Direct workflow children retain this session even after reparenting;
 # their separate process groups still support independent cancellation.
 found={owner.pid} if owner.poll() is None else set()
 for pid in current:
  try:
   if os.getsid(pid)==owner.pid:found.add(pid)
  except OSError:pass
 while True:
  more={pid for pid,(parent,_) in current.items() if parent in found}-found
  if not more:break
  found.update(more)
 for pid in found:
  if pid in current and pid not in known and not current[pid][1].endswith('/bin/ps -axo pid,ppid,lstart,command'):
   identity=process_identity(pid)
   if identity is not None:known[pid]=identity
 return current
def kill_owned(current):
 for pid,identity in known.items():
  if pid!=os.getpid() and alive(identity):
   try:os.kill(pid,signal.SIGKILL);forced.append(pid)
   except ProcessLookupError:pass
def interrupted(*_):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
 with (root/'owner-console.log').open('xb') as log:
  owner=subprocess.Popen([meta['python'],'-u',str(root/'workflow.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write('launch.json',{'watchdog_pid':os.getpid(),'owner_pid':owner.pid,'started':started,'original_launch_at':meta['original_launch_at'],'absolute_deadline_at':meta['absolute_deadline_at'],'remaining_seconds':remaining,'remote_root':str(root),'credential_environment':meta.get('credential_environment'),'stop_lead_seconds':grace,'policy':'No retry, resume, top-up, extension, or follow-on run.'})
 while owner.poll() is None:
  current=track();now=time.monotonic()
  if now>=end-grace and not signalled:
   reason='absolute_deadline';signalled=True
   try:os.kill(owner.pid,signal.SIGTERM)
   except ProcessLookupError:pass
  if now>=end:
   kill_owned(current);break
  time.sleep(min(.5,max(.01,end-now)))
except KeyboardInterrupt:
 reason='supervisor_interrupted'
 if owner is not None:
  try:os.kill(owner.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:owner.wait(timeout=min(10,max(.1,end-time.monotonic())))
  except subprocess.TimeoutExpired:pass
finally:
 # Clean only exact identities observed under this owner. Separate sessions do
 # not escape the deadline, and PID reuse cannot target an unrelated process.
 current=track();kill_owned(current)
 if owner is not None:
  try:owner.wait(timeout=2)
  except subprocess.TimeoutExpired:pass
 try:current=rows()
 except (OSError,subprocess.SubprocessError):current={};tracking_errors+=1
 survivors=[pid for pid,identity in known.items() if alive(identity)]
 receipt={'pid':owner.pid if owner else None,'watchdog_pid':os.getpid(),'returncode':owner.returncode if owner else None,'stop_reason':reason,'elapsed_seconds':time.monotonic()-clock,'execution_seconds':time.monotonic()-clock,'execution_limit_seconds':remaining,'absolute_deadline_at':meta['absolute_deadline_at'],'finished':time.time(),'forced_killed_pids':forced,'owned_surviving_pids':survivors,'owned_identities':known,'tracking_errors':tracking_errors,'cleanup_verified':tracking_errors==0 and not survivors}
 write('owner-limit.json',receipt)
 # Publish terminal status even if the workflow could not enter its own
 # exception handler. Readers also reconcile from owner-limit.json if this
 # best-effort metadata repair cannot be written.
 try:latest=json.loads((root/'instance.json').read_text())
 except (OSError,ValueError):latest=meta
 terminal=effective_run_state(latest,receipt)
 best_effort_write(root/'instance.json',terminal)
raise SystemExit(0 if owner is not None and owner.returncode==0 and reason is None and receipt['cleanup_verified'] else 1)
