"""Exercise the configuration-to-capture contract, not just audit helpers."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.episode import orchestrator as orchestrator_module
from shellsmith.episode.acceptance_audit import audit_rollout
from shellsmith.episode.orchestrator import EpisodeOrchestrator, EpisodePaths, _thinking_mode_for_effort
from shellsmith.episode.protocol_integrity import audit_protocol_integrity, model_protocol_requests_from_rollout_rows
from shellsmith.policy.reasoning import thinking_mode_for_effort
from shellsmith.policy.sampler import PlayerRequestError, thinking_requested
from shellsmith.train.config import load_train_config
from shellsmith.train.rollout_stack import episode_args
from test_acceptance_audit import _rollout, _write


def _protocol(root):
    return json.loads((root / 'scenario/model_protocol.json').read_text())


def _save_protocol(root, protocol):
    _write(root / 'scenario/model_protocol.json', protocol)


def _vat(root):
    return [json.loads(line) for line in (root / 'logs/vat.jsonl').read_text().splitlines()]


def _save_vat(root, rows):
    (root / 'logs/vat.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.mark.parametrize('effort', ['low', 'medium', 'xhigh', 'high', 'deep', 'thinking', 'on', 'enable_thinking', ' Medium '])
def test_reasoning_enabled_labels_agree_across_all_boundaries(effort):
    assert thinking_mode_for_effort(effort) == _thinking_mode_for_effort(effort) == 'thinking'
    for payload in ({'reasoning_effort': effort}, {'reasoningEffort': effort}, {'reasoning': {'effort': effort}}):
        assert thinking_requested(payload)


@pytest.mark.parametrize('effort', ['off', 'none', 'minimal', 'disable_thinking', ' OFF '])
def test_explicit_disabled_labels_agree_across_all_boundaries(effort):
    assert thinking_mode_for_effort(effort) == _thinking_mode_for_effort(effort) == 'disable_thinking'
    assert not thinking_requested({'reasoning_effort': effort})


@pytest.mark.parametrize('effort', ['medum', '', 1, True, {}])
def test_unknown_effort_never_silently_disables_reasoning(effort):
    with pytest.raises(ValueError):
        _thinking_mode_for_effort(effort)
    with pytest.raises(PlayerRequestError):
        thinking_requested({'reasoning_effort': effort})


def test_missing_request_effort_is_plain_but_legacy_episode_default_is_thinking():
    assert not thinking_requested({})
    assert _thinking_mode_for_effort(None) == 'thinking'


@pytest.mark.parametrize('effort,limit', [('low', 2048), ('medium', 4096), ('xhigh', 8192)])
def test_episode_setup_pins_resolved_mode_and_actual_budgets(tmp_path, monkeypatch, effort, limit):
    config = load_train_config(Path(__file__).resolve().parents[1] / 'examples/studio-micro-train.toml')
    config = replace(config, studio=replace(config.studio, reasoning_effort=effort))
    args = episode_args(config)
    assert args.player_reasoning_tokens == limit
    assert args.player_response_tokens == 32768
    owner = object.__new__(EpisodeOrchestrator)
    owner.paths = EpisodePaths.from_root(tmp_path / 'prepared')
    owner.vat_process = SimpleNamespace(config=SimpleNamespace(workspace=SimpleNamespace(assert_empty=lambda: None)))
    owner.audit_metadata = {'player_renderer': f'qwen3_8_{effort}_reasoning', 'player_thinking': effort,
                            'player_reasoning_tokens': args.player_reasoning_tokens,
                            'player_response_tokens': args.player_response_tokens}
    owner.prompt_root = tmp_path
    owner.world_prompt_profile = owner.prompt_inputs = None
    owner.spec = SimpleNamespace(markdown='fixture world', id='fixture', content_hash=lambda: 'hash')
    owner.episode_id = 'protocol-fixture'
    monkeypatch.setattr(orchestrator_module, 'render_world_agents', lambda **kwargs: 'fixture instructions')
    asyncio.run(owner.prepare_episode())
    protocol = _protocol(owner.paths.root)
    assert protocol['expected_thinking_mode'] == 'thinking'
    assert protocol['thinking_token_budget'] == limit
    assert protocol['response_token_budget'] == 32768


@pytest.mark.parametrize('effort', ['low', 'medium', 'xhigh'])
@pytest.mark.parametrize('explicit_stub', [False, True])
def test_effort_mapping_is_used_by_actual_acceptance_gate(tmp_path, effort, explicit_stub):
    root = _rollout(tmp_path)
    config = json.loads((root / 'run_config.json').read_text())
    config['player_thinking'] = effort
    _write(root / 'run_config.json', config)
    protocol = _protocol(root)
    if not explicit_stub:
        protocol.pop('expected_thinking_mode')
    _save_protocol(root, protocol)
    report = audit_rollout(root)
    assert report['accepted_for_training'], report['model_protocol_integrity']['problems']
    assert report['model_protocol_integrity']['expected_thinking_mode'] == 'thinking'


def test_stale_pinned_mode_conflicts_with_configuration_instead_of_overriding_it(tmp_path):
    root = _rollout(tmp_path)
    config = json.loads((root / 'run_config.json').read_text())
    config['player_thinking'] = 'medium'
    _write(root / 'run_config.json', config)
    protocol = _protocol(root)
    protocol['expected_thinking_mode'] = 'disable_thinking'
    _save_protocol(root, protocol)
    report = audit_rollout(root)
    assert not report['accepted_for_training']
    assert any('conflicts with run_config.player_thinking' in p for p in report['model_protocol_integrity']['problems'])


def test_unknown_recorded_effort_is_a_loud_audit_failure(tmp_path):
    root = _rollout(tmp_path)
    config = json.loads((root / 'run_config.json').read_text())
    config['player_thinking'] = 'medum'
    _write(root / 'run_config.json', config)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any('player_thinking is invalid' in p for p in report['problems'])


@pytest.mark.parametrize('termination', ['stop', 'stop_sequence', 'eos', 'clean'])
def test_native_and_tinker_clean_stops_are_not_anomalies(tmp_path, termination):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    protocol['requests'][0]['parse_termination'] = termination
    _save_protocol(root, protocol)
    report = audit_protocol_integrity(root)
    assert report['passed'] and not report['warnings']


def test_length_limit_remains_trainable_negative_behavior(tmp_path):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    protocol['requests'][0]['parse_termination'] = 'length'
    _save_protocol(root, protocol)
    report = audit_rollout(root)
    assert report['accepted_for_training']
    assert any('not clean' in p for p in report['model_protocol_integrity']['warnings'])


def test_studio_saved_records_recover_an_empty_protocol_stub(tmp_path):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    (root / 'logs/policy_rollout_records.jsonl').write_text(json.dumps(protocol['requests'][0]) + '\n')
    protocol['requests'] = []
    _save_protocol(root, protocol)
    assert audit_rollout(root)['accepted_for_training']


@pytest.mark.parametrize('field,value', [('tool_calls', {}), ('assistant_content_types', 1),
    ('thinking_mode', {}), ('sampled_token_count', -1), ('unparsed_tool_call_count', True),
    ('compaction_summary', 'true'), ('request_id', ''), ('renderer_name', None)])
def test_malformed_capture_metadata_cannot_be_silently_coerced(tmp_path, field, value):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    protocol['requests'][0][field] = value
    _save_protocol(root, protocol)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any(field in p for p in report['problems'])
    with pytest.raises(RuntimeError, match='Malformed rollout protocol evidence'):
        model_protocol_requests_from_rollout_rows(protocol['requests'])


@pytest.mark.parametrize('field', ['renderer_name', 'thinking_mode'])
def test_missing_mode_or_renderer_does_not_bypass_pinned_expectation(tmp_path, field):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    protocol['requests'][0].pop(field)
    _save_protocol(root, protocol)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any('mismatch' in p for p in report['problems'])


def test_duplicate_request_ids_are_rejected_even_when_counts_match(tmp_path):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    # Add a second text sample with the same request id, otherwise fully aligned.
    second = dict(protocol['requests'][0], tool_calls=[], assistant_text='Done.')
    protocol['requests'].append(second)
    _save_protocol(root, protocol)
    rows = _vat(root)
    rows.append({'message': {'type': 'agent_event', 'event_type': 'message_end',
                            'payload': {'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Done.'}]}}}})
    _save_vat(root, rows)
    report = audit_protocol_integrity(root)
    assert any('duplicate model request ID' in p for p in report['problems'])


@pytest.mark.parametrize('mutation', ['id', 'name', 'arguments', 'text'])
def test_proxy_and_pi_projection_must_match_not_just_their_counts(tmp_path, mutation):
    root = _rollout(tmp_path)
    rows = _vat(root)
    content = rows[1]['message']['payload']['message']['content']
    if mutation == 'text':
        content.append({'type': 'text', 'text': 'Injected content'})
    else:
        content[1][mutation] = {'command': 'echo wrong'} if mutation == 'arguments' else 'wrong'
    _save_vat(root, rows)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any('Pi assistant projection' in p for p in report['problems'])


@pytest.mark.parametrize('mutation', ['missing_result', 'duplicate_result', 'orphan_result', 'duplicate_world_id', 'bad_turn_name', 'forged_rejection', 'fake_compaction'])
def test_tool_and_compaction_corruption_remains_rejected(tmp_path, mutation):
    root = _rollout(tmp_path)
    rows, protocol = _vat(root), _protocol(root)
    if mutation == 'missing_result':
        rows.pop()
    elif mutation == 'duplicate_result':
        rows.append(rows[-1])
    elif mutation == 'orphan_result':
        rows.append({'message': {'type': 'tool_response', 'tool_call_id': 'orphan', 'content': 'bad'}})
    elif mutation == 'duplicate_world_id':
        path = root / 'world/turns/000000/request.json'
        request = json.loads(path.read_text())
        request['actions'] = [request['action'], request['action']]
        _write(path, request)
    elif mutation == 'bad_turn_name':
        (root / 'world/turns/000000').rename(root / 'world/turns/invalid')
    elif mutation == 'forged_rejection':
        protocol['requests'][0]['tool_calls'][0]['harness_rejected'] = True
    elif mutation == 'fake_compaction':
        # A compaction flag must not hide a world action from correlation.
        protocol['requests'][0]['compaction_summary'] = True
    _save_vat(root, rows)
    _save_protocol(root, protocol)
    report = audit_protocol_integrity(root)
    assert not report['passed'], mutation


def test_compaction_summary_requires_actual_pi_compaction_evidence(tmp_path):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    summary = dict(protocol['requests'][0], request_id='summary', compaction_summary=True,
                   tool_calls=[], assistant_text='Invented summary')
    protocol['requests'].append(summary)
    _save_protocol(root, protocol)
    report = audit_protocol_integrity(root)
    assert any('sample/compaction journal alignment failed' in p for p in report['problems'])


def test_schema_rejected_model_tool_call_stays_trainable_with_one_pi_error(tmp_path):
    root = _rollout(tmp_path)
    bad = {'id': 'invalid-call', 'name': 'bash', 'arguments': {'command': 17}}
    protocol = _protocol(root)
    protocol['requests'][0]['tool_calls'].append(bad)
    _save_protocol(root, protocol)
    rows = _vat(root)
    rows[1]['message']['payload']['message']['content'].append({'type': 'toolCall', **bad})
    rows.extend([
        {'message': {'type': 'tool_request', 'tool_call_id': bad['id'], 'tool_name': bad['name'],
                     'arguments': bad['arguments'], 'harness_rejected': True}},
        {'message': {'type': 'tool_response', 'tool_call_id': bad['id'], 'content': 'command must be a string', 'is_error': True}},
    ])
    _save_vat(root, rows)
    report = audit_rollout(root)
    assert report['accepted_for_training'], report['model_protocol_integrity']['problems']
    rows.pop()
    _save_vat(root, rows)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any("'invalid-call' was returned to Pi 0 times" in p for p in report['problems'])


def test_world_error_does_not_exempt_tool_argument_correlation(tmp_path):
    root = _rollout(tmp_path)
    rows = _vat(root)
    rows[-1]['message']['is_error'] = True
    _save_vat(root, rows)
    response_path = root / 'world/turns/000000/response.json'
    response = json.loads(response_path.read_text())
    response['tool_response']['is_error'] = True
    _write(response_path, response)
    assert audit_rollout(root)['accepted_for_training']
    protocol = _protocol(root)
    protocol['requests'][0]['tool_calls'][0]['arguments'] = {'command': 'wrong'}
    _save_protocol(root, protocol)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any('tool arguments mismatch' in p for p in report['problems'])


def test_host_path_projection_matches_before_world_virtualization(tmp_path):
    root = _rollout(tmp_path)
    host = '/tmp/shellsmith-fixture/workspace'
    command = f'cat {host}/README.md'
    protocol = _protocol(root)
    protocol['requests'][0]['tool_calls'][0]['arguments'] = {'command': command}
    _save_protocol(root, protocol)
    rows = _vat(root)
    rows[0]['message']['workspace_path'] = host
    rows[1]['message']['payload']['message']['content'][1]['arguments'] = {'command': command}
    rows[2]['message']['arguments'] = {'command': command}
    _save_vat(root, rows)
    request_path = root / 'world/turns/000000/request.json'
    request = json.loads(request_path.read_text())
    request['action']['arguments'] = {'command': 'cat /workspace/README.md'}
    _write(request_path, request)
    report = audit_rollout(root)
    assert report['accepted_for_training'], report['model_protocol_integrity']['problems']


def test_conflicting_fallback_record_files_are_not_arbitrarily_selected(tmp_path):
    root = _rollout(tmp_path)
    protocol = _protocol(root)
    row = protocol['requests'][0]
    (root / 'logs/policy_rollout_records.jsonl').write_text(json.dumps(row) + '\n')
    (root / 'logs/tinker_rollout_records.jsonl').write_text(json.dumps(dict(row, request_id='other')) + '\n')
    protocol['requests'] = []
    _save_protocol(root, protocol)
    with pytest.raises(RuntimeError, match='Conflicting rollout record files'):
        audit_protocol_integrity(root)


def test_symlinked_protocol_evidence_is_refused(tmp_path):
    root = _rollout(tmp_path)
    path = root / 'scenario/model_protocol.json'
    moved = root / 'original-model-protocol.json'
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(RuntimeError, match='forbidden symlink'):
        audit_protocol_integrity(root)


@pytest.mark.parametrize('mutation', ['arguments', 'name', 'duplicate', 'orphan'])
def test_harness_tool_requests_match_the_projected_assistant(tmp_path, mutation):
    root = _rollout(tmp_path)
    rows = _vat(root)
    if mutation == 'arguments':
        rows[2]['message']['arguments'] = {'command': 'injected command'}
    elif mutation == 'name':
        rows[2]['message']['tool_name'] = 'wrong'
    elif mutation == 'duplicate':
        rows.append(rows[2])
    else:
        extra = json.loads(json.dumps(rows[2]))
        extra['message']['tool_call_id'] = 'orphan'
        rows.append(extra)
    _save_vat(root, rows)
    report = audit_protocol_integrity(root)
    assert not report['passed']
    assert any('harness request' in p or 'harness tool request' in p for p in report['problems'])
