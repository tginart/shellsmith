"""Tiny real-27B checks, isolated from the four authorized world attempts."""
import asyncio,json,math,time
from dataclasses import replace
from pathlib import Path
from studio_compute.client import Client
from shellsmith.train.config import load_train_config
from shellsmith.train.studio import StudioBackend
from studio_compute.protocol import validate_request
from shellsmith.mlx.generation import completion_budget


def run(root, meta, start_service, stop_service, write):
    report={'status':'running','scope':'Tiny native compute preflight; no world attempts','started':time.time(),'checks':{},'operations':[]}
    until=min(meta['absolute_deadline_at'],time.time()+600)
    def save():write(root/'native-preflight.json',report)
    def call(client,kind,body,name):
        remaining=until-time.time()
        if remaining<=0:raise TimeoutError('Tiny native preflight exceeded 600 seconds')
        client.timeout=min(180,remaining)
        began=time.time();result=client.call(kind,body,request_id=meta['run_id']+':preflight:'+name)
        report['operations'].append({'id':name,'kind':kind,'seconds':time.time()-began,'result':result});save();return result
    config=load_train_config(root/'train.toml')
    config=replace(config,player=replace(config.player,prompt_timeout_seconds=180),run=replace(config.run,log_dir=str(root/'preflight-coordinator')),
                   studio=replace(config.studio,training_client=meta['run_id']+'-preflight',operation_timeout_seconds=180))
    backend=StudioBackend(config)
    control=Client(meta['endpoint'],meta['run_id']+'-preflight-control',timeout=180)
    try:
        asyncio.run(backend.initialize());policy=asyncio.run(backend.publish_policy(revision=0))
        cap=backend.capabilities;identity=cap['identity']
        assert cap['precision']['activations']==cap['precision']['lora']==cap['precision']['training_kv']=='bf16'
        assert cap['precision']['optimizer']==cap['precision']['recurrent_state']=='fp32'
        assert cap['host_headroom_bytes']==16*1024**3 and cap['allocator_budget_bytes']==240*1024**3
        before=call(backend.client,'inspect',{'hash_state':True,'hash_base':True},'before')
        assert before['frozen_base_hash']==meta['expected_base_hash']
        payload={'messages':[{'role':'user','content':'Call shell_command once with command exactly printf PREFLIGHT. Do not provide a text answer.'}],
                 'tools':[{'type':'function','function':{'name':'shell_command','description':'Run one shell command',
                  'parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command'],'additionalProperties':False}}}],
                 'reasoning_effort':'medium'}
        record,assistant=backend.sampler.sample(payload,request_id='native-preflight-canary',session_id='native-preflight-canary')
        backend.sampler.release('native-preflight-canary')
        calls=assistant.get('tool_calls') or []
        assert len(calls)==1 and calls[0]['function']['name']=='shell_command',assistant
        assert json.loads(calls[0]['function']['arguments'])=={'command':'printf PREFLIGHT'},assistant
        assert record.protocol_row['thinking_mode']=='thinking' and record.protocol_row['renderer_name']=='qwen3_8_medium_reasoning'
        report['checks']['medium_two_pass_tool_template']={'passed':True,'prompt_tokens':record.prompt_tokens,
            'completion_tokens':len(record.sampled_tokens),'reasoning_present':bool(assistant.get('reasoning_content')),
            'tool_calls':calls,'request_only_completion_cap':None,'configured_reasoning_tokens':4096,'configured_response_tokens':config.studio.response_tokens,
            'protocol':record.protocol_row,'tool_executed':False};save()
        # Admission only, not a full-context GPU capacity claim.
        assert record.protocol_row['response_limit']==config.studio.response_tokens==32768
        reserve=completion_budget(config.studio)
        body={'policy':policy,'prompt_token_ids':[1]*(262144-reserve),'max_tokens':reserve}
        validate_request('sample',body,cap)
        try:validate_request('sample',{**body,'prompt_token_ids':body['prompt_token_ids']+[1]},cap)
        except ValueError:pass
        else:raise AssertionError('Overflow was admitted')
        report['checks']['context_contract']={'passed':True,'total_window':262144,'completion_reserve':reserve,
            'exact_prompt_admitted':262144-reserve,'overflow_rejected':True,'gpu_full_context_test':False};save()
        tokenizer=backend.renderer.tokenizer
        unit=tokenizer.encode(' item',add_special_tokens=False)
        prompt=(unit*64)[:64]
        sample={'policy':policy,'prompt_token_ids':prompt,'max_tokens':8,'seed':31,'cache_precision':'bf16'}
        cold=call(backend.client,'sample',{**sample,'use_prefix_cache':False},'cold')
        call(backend.client,'sample',sample,'prefix-store')
        warm=call(backend.client,'sample',sample,'prefix-warm')
        assert warm['prefix_cache_tokens']==len(prompt) and warm['sampled_tokens']==cold['sampled_tokens']
        assert max(abs(x-y) for x,y in zip(warm['behavior_logprobs'],cold['behavior_logprobs']))<1e-5
        report['checks']['prefix_reuse']={'passed':True,'prompt_tokens':len(prompt),'reused_tokens':warm['prefix_cache_tokens']};save()
        targets=(tokenizer.encode(' Four.',add_special_tokens=False)*8)[:8]
        scored=call(backend.client,'logprobs',{'policy':policy,'prompt_token_ids':prompt,'target_token_ids':targets},'score')
        sequences=[{'prompt_token_ids':prompt,'sampled_tokens':targets,'behavior_logprobs':scored['behavior_logprobs'],
                    'advantages':[.1/len(targets)]*len(targets),'loss_mask':[1.]*len(targets)}]
        forward=call(backend.client,'forward_backward',{'loss':'cispo','reduction':'sum','clip_low':0.,'clip_high':4.,'sequences':sequences},'backward')
        optimizer=call(backend.client,'optim_step',{'learning_rate':.00005,'backward_operation':meta['run_id']+':preflight:backward'},'optimizer')
        assert math.isfinite(forward['forward_metrics']['loss']) and math.isfinite(optimizer['optim_metrics']['gradient_norm'])
        assert optimizer['optim_metrics']['optimizer_steps']==1
        published=call(backend.client,'publish',{'revision':1},'publication')
        saved=call(backend.client,'save',{},'save')
        after=call(backend.client,'inspect',{'hash_state':True,'hash_base':True},'after')
        assert after['frozen_base_hash']==before['frozen_base_hash']
        assert after['state_hashes']['adapter']!=before['state_hashes']['adapter']
        call(control,'restore',{'checkpoint':saved['checkpoint'],'expected_identity':identity},'control-restore')
        cloned=call(control,'inspect',{'hash_state':True},'control-inspect')
        assert cloned['state_hashes']==after['state_hashes']
        next_body={'policy':published['policy'],'prompt_token_ids':prompt,'max_tokens':8,'cache_precision':'bf16'}
        expected=call(control,'sample',next_body,'expected-next')
        first_pid=report['service_pid_before_restart']=meta['service_pid']
        stop_service();newcap=start_service()
        assert meta['service_pid']!=first_pid and newcap['identity']==identity
        restored=call(backend.client,'inspect',{'hash_state':True,'hash_base':True},'restored')
        assert restored['state_hashes']==after['state_hashes'] and restored['frozen_base_hash']==before['frozen_base_hash']
        actual=call(backend.client,'sample',next_body,'actual-next')
        assert actual['sampled_tokens']==expected['sampled_tokens'] and actual['behavior_logprobs']==expected['behavior_logprobs']
        report['checks']['training_durability_recovery']={'passed':True,'prompt_tokens':len(prompt),'completion_targets':len(targets),
            'finite_loss':forward['forward_metrics']['loss'],'finite_gradient_norm':optimizer['optim_metrics']['gradient_norm'],
            'checkpoint':saved['checkpoint'],'published_policy':published['policy'],'frozen_base_unchanged':True,
            'exact_adapter_optimizer_rng_restore':True,'exact_next_sample':True,'fresh_process_pid':meta['service_pid']}
        report.update(status='passed',finished=time.time(),precision=cap['precision'],before=before,after=after,
                      final=restored,source_sha256=meta['source_sha256'],merged_commit=meta['merged_commit'])
        save();return before
    except BaseException as error:
        report.update(status='failed',finished=time.time(),error=type(error).__name__+': '+str(error));save();raise
    finally:
        backend.close()
        # Deliberate raw RPC cleanup; this client was never owned by Shellsmith.
        try:call(control,'suspend',{},'control-suspend')
        except Exception:pass
