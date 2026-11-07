/* The dashboard intentionally has no build step or framework. Durable artifact JSON
 * is the state; filtering and accordion rendering happen entirely in the browser. */
const state={runs:[],filtered:[],selected:null};
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(value,digits=2)=>typeof value==="number"?value.toFixed(digits):"—";
const scoreClass=value=>value>=.8?"good":value>=.5?"mid":"low";
const pretty=value=>value===null||value===undefined?null:JSON.stringify(value,null,2);
const pill=(label,kind="")=>`<span class="pill ${kind}">${esc(label)}</span>`;

function showError(message){const el=$("error");el.textContent=message;el.classList.toggle("hidden",!message)}
function setOptions(id,values){const el=$(id),first=el.options[0];el.innerHTML="";el.append(first);[...new Set(values)].sort().forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;el.append(o)})}
function metrics(){
  const complete=state.filtered.filter(r=>typeof r.reward==="number");
  const avg=complete.length?complete.reduce((a,r)=>a+r.reward,0)/complete.length:null;
  const withQ=state.filtered.filter(r=>typeof r.mean_quality==="number");
  const withE=state.filtered.filter(r=>typeof r.mean_efficiency==="number");
  const avgQ=withQ.length?withQ.reduce((a,r)=>a+r.mean_quality,0)/withQ.length:null;
  const avgE=withE.length?withE.reduce((a,r)=>a+r.mean_efficiency,0)/withE.length:null;
  const cards=[
    ["Rollouts",state.filtered.length,"visible"],
    ["Mean reward",fmt(avg,3),`${complete.length} judged`],
    ["Mean quality",fmt(avgQ,3),`${withQ.length} assessed`],
    ["Mean efficiency",fmt(avgE,3),`${withE.length} assessed`],
    ["Trainable",state.filtered.filter(r=>r.trainable===true).length,"final signal"],
    ["Integrity pass",state.filtered.filter(r=>r.integrity_passed===true).length,"GM integrity clean"],
    ["Accepted",state.filtered.filter(r=>r.accepted_for_training===true).length,"audit gate"],
    ["Total actions",state.filtered.reduce((a,r)=>a+(r.action_count||0),0),"visible runs"]
  ];
  $("metrics").innerHTML=cards.map(c=>`<article class="metric"><div class="label">${c[0]}</div><div class="value">${c[1]}</div><div class="sub">${c[2]}</div></article>`).join("")
}
function endPill(r){
  if(r.end_kind==="agent_terminal")return pill("agent terminal","warn");
  if(r.end_kind==="world_terminal")return pill("world terminal","ok");
  if(r.end_kind==="budget_terminal")return pill("budget terminal","warn");
  if(r.end_kind==="vat_terminal")return pill("vat terminal","warn");
  return "";
}
// Campaign-level states that are NOT ordinary low-reward rollouts. Without
// these chips a never-started slot and a discarded fallback group look exactly
// like a normal failing episode.
function outcomePills(r){
  let out="";
  if(r.episode_not_started===true)out+=pill("never started","bad");
  if(r.discarded_by_provider_fallback===true)out+=pill("discarded","bad");
  if(r.gm_fallback_applied===true)out+=pill("backup gm","warn");
  return out
}
function warningsHtml(s){
  const items=s.warnings||[];
  if(!items.length)return"";
  return`<div class="warnings">${items.map(w=>`<div class="warn-row">${esc(w)}</div>`).join("")}</div>`
}
function matchesStateFilter(r,mode){
  if(!mode)return true;
  if(mode==="trainable")return r.trainable===true;
  if(mode==="untrainable")return r.trainable===false;
  if(mode==="integrity-pass")return r.integrity_passed===true;
  if(mode==="integrity-fail")return r.integrity_passed===false;
  if(mode==="agent-terminal")return r.end_kind==="agent_terminal";
  if(mode==="dangling-final")return r.has_dangling_agent_final===true;
  if(mode==="world-terminal")return r.end_kind==="world_terminal";
  if(mode==="budget-terminal")return r.end_kind==="budget_terminal";
  if(mode==="never-started")return r.episode_not_started===true;
  if(mode==="discarded")return r.discarded_by_provider_fallback===true;
  if(mode==="fallback-gm")return r.gm_fallback_applied===true;
  return true;
}
function applyFilters(){const q=$("search").value.toLowerCase(),campaign=$("campaignFilter").value,run=$("runFilter").value,task=$("taskFilter").value,model=$("modelFilter").value,mode=$("stateFilter").value,min=Number($("rewardFilter").value);$("rewardValue").value=min.toFixed(2);state.filtered=state.runs.filter(r=>{const hay=[r.id,r.run_dir,r.campaign,r.batch,r.task,r.player_model,r.episode_id,r.agent_final_preview,r.end_label].join(" ").toLowerCase();return(!q||hay.includes(q))&&(!campaign||r.campaign===campaign)&&(!run||r.run_dir===run)&&(!task||r.task===task)&&(!model||r.player_model===model)&&(typeof r.reward!=="number"||r.reward>=min)&&matchesStateFilter(r,mode)});metrics();renderCatalog()}
function renderCatalog(){const mode=$("groupBy").value,groups={};state.filtered.forEach(r=>{let key=r.task;if(mode==="campaign")key=r.campaign||"Standalone / no campaign";if(mode==="model")key=r.player_model;if(mode==="batch")key=r.batch||"Standalone";if(mode==="run")key=r.run_dir;if(mode==="reward")key=typeof r.reward!=="number"?"Pending":r.reward>=.8?"High · 0.80–1.00":r.reward>=.5?"Medium · 0.50–0.79":"Low · 0.00–0.49";(groups[key]??=[]).push(r)});$("visibleCount").textContent=`${state.filtered.length} visible`;$("runGroups").innerHTML=Object.entries(groups).map(([label,runs])=>`<div class="group-title">${esc(label)} · ${runs.length}</div>${runs.map(runCard).join("")}`).join("")||`<div class="empty-state"><p>No matching runs.</p></div>`;document.querySelectorAll(".run").forEach(el=>el.onclick=()=>selectRun(el.dataset.id))}
function statusPill(r){
  // Episode status can be "untrainable", which duplicates the trainable boolean pill.
  // Only surface lifecycle anomalies the boolean pills do not already cover.
  const status=String(r.status||"").toLowerCase();
  if(!status||status==="complete"||status==="untrainable")return"";
  if(status==="failed")return pill("failed","bad");
  return pill(status,"warn");
}
function runCard(r){
  const integrity=r.integrity_passed===true?pill("integrity","ok"):r.integrity_passed===false?pill("integrity","bad"):pill("no audit");
  const train=r.trainable===true?pill("trainable","ok"):r.trainable===false?pill("untrainable","bad"):pill("pending","warn");
  const campaignShort=(r.campaign||"").split("/").pop();
  return`<button class="run ${state.selected===r.id?"active":""}" data-id="${esc(r.id)}"><div class="run-top"><span class="run-name">${esc(r.episode_id)}</span><span class="score-stack"><span class="reward" title="final reward">R ${fmt(r.reward,3)}</span><span class="score ${scoreClass(r.mean_quality)}" title="mean reasoning quality">Q ${fmt(r.mean_quality)}</span><span class="score ${scoreClass(r.mean_efficiency)}" title="mean reasoning efficiency">E ${fmt(r.mean_efficiency)}</span></span></div><div class="run-meta">${esc(r.task)} · ${esc(r.gm_model||r.player_model)} · ${r.action_count??"—"} actions${campaignShort?` · ${esc(campaignShort)}`:""}</div><div class="badges">${train}${integrity}${endPill(r)}${outcomePills(r)}${statusPill(r)}</div></button>`
}
async function selectRun(id){state.selected=id;renderCatalog();const detail=$("detail");detail.className="panel detail";detail.innerHTML='<div class="spinner"></div>';try{const response=await fetch(`/api/runs/${id.split("/").map(encodeURIComponent).join("/")}`),contentType=response.headers.get("content-type")||"",data=contentType.includes("application/json")?await response.json():null;if(!response.ok)throw new Error(data?.error||`Rollout detail request failed (${response.status})`);if(!data)throw new Error("Rollout detail endpoint returned a non-JSON response");renderDetail(data)}catch(error){detail.innerHTML=`<div class="error">${esc(error.message)}</div>`}}
function datum(label,value){return`<div class="datum"><label>${esc(label)}</label><span>${esc(value??"—")}</span></div>`}
function danglingFinalHtml(s){
  // Only agent-terminal ends are "dangling". World/budget stops may leave leftover
  // assistant text after terminate=true; that is not why the episode stopped.
  if(s.end_kind!=="agent_terminal")return"";
  const text=s.agent_final_response;
  const chips=`<div class="badges">${endPill(s)}${text?pill("dangling final","warn"):pill("empty final","bad")}</div>`;
  if(!text){
    return`<div class="dangling-final"><h3>Dangling final agent response</h3>${chips}<p class="muted">Episode ended on an agent-terminal stop, but the recorded final assistant text was empty (e.g. after compaction).</p></div>`;
  }
  return`<div class="dangling-final"><h3>Dangling final agent response</h3>${chips}<p class="muted">Player's last assistant message had no structured tool call, so Pi settled and GM never saw another turn.</p><pre class="dangling-final-text">${esc(text)}</pre></div>`;
}
function reasoningByType(reasoning,type){
  return(reasoning||[]).filter(r=>r&&r.type===type&&r.text).map(r=>String(r.text).trim()).filter(Boolean);
}
function initialUserHtml(text){
  // Kickoff Pi user message (task brief) from turn 0 trajectory_delta.
  if(!text)return`<div class="initial-user"><h3>First user message</h3><p class="muted">No kickoff user message found in turn #000 trajectory_delta or scenario/player_agent_prompt.md.</p></div>`;
  return`<div class="initial-user"><h3>First user message</h3><p class="muted">Kickoff prompt that started Player (Pi user role).</p><pre class="initial-user-text">${esc(text)}</pre></div>`;
}
function kickoffAgentsHtml(text){
  // Durable copy of the vat AGENTS.md GM wrote before Player started.
  if(!text)return`<div class="kickoff-agents"><h3>Kickoff AGENTS.md</h3><p class="muted">No scenario/vat_AGENTS.md found for this episode.</p></div>`;
  return`<div class="kickoff-agents"><h3>Kickoff AGENTS.md</h3><p class="muted">Vat workspace instructions Player saw at boot (<code>scenario/vat_AGENTS.md</code>).</p><pre class="kickoff-agents-text">${esc(text)}</pre></div>`;
}
function generationDiagnosticsHtml(data){
  const rows=(Array.isArray(data?.requests)?data.requests:[]).filter(row=>row&&typeof row==="object");
  if(!rows.length)return"";
  return`<section class="generation-diagnostics"><h3>Generation limits and errors</h3><p class="muted">Includes rejected responses and recoverable reasoning limits. A reasoning limit closes thinking and leaves response generation enabled.</p>${rows.map(row=>block(row.request_id||"Request",[...(Array.isArray(row.generation_notices)?row.generation_notices:[]).filter(n=>n&&typeof n==="object").map(n=>`${n.code}: ${n.message}`),row.protocol_error?`Protocol error: ${row.protocol_error}`:""].filter(Boolean).join("\n"))).join("")}</section>`;
}
function renderDetail(data){
  const s=data.summary,j=data.terminal_judgment||{},a=data.acceptance_audit||{};
  const turns=data.turns||[];
  $("detail").innerHTML=`<header class="run-heading"><div><h2>${esc(s.episode_id)}</h2><p>${esc(s.task)} · ${esc(s.player_model)} · GM: ${esc(s.gm_model)}</p><div class="badges">${endPill(s)}${outcomePills(s)}${s.has_dangling_agent_final?pill("dangling final","warn"):""}</div></div><div class="big-scores"><div class="big-reward">${fmt(s.reward,3)}<small>final reward</small></div><div class="big-reward">${fmt(s.mean_quality,3)}<small>mean quality</small></div><div class="big-reward">${fmt(s.mean_efficiency,3)}<small>mean efficiency</small></div></div></header>${warningsHtml(s)}${generationDiagnosticsHtml(data.generation_diagnostics)}${data.artifact_errors.length?`<div class="error">${data.artifact_errors.map(esc).join("<br>")}</div>`:""}<div class="detail-grid">${datum("Status",s.status)}${datum("End kind",s.end_label||"—")}${datum("Completion",fmt(s.completion_percentage,0)+"%")}${datum("Trainable",s.trainable)}${datum("Integrity",s.integrity_passed)}${datum("Actions",s.action_count)}${datum("Accepted",s.accepted_for_training)}${datum("Mean quality",fmt(s.mean_quality,3))}${datum("Mean efficiency",fmt(s.mean_efficiency,3))}${datum("Campaign",s.campaign)}${datum("Batch",s.batch)}${datum("Game Master",s.gm_model)}${datum("Fell back from",s.original_gm_model)}${datum("Policy",s.policy_kind)}${datum("Policy ref",s.policy_ref)}${datum("RFT updates",s.policy_rft_updates_completed)}</div><nav class="tabs"><button class="tab active" data-tab="trajectory">Trajectory</button><button class="tab" data-tab="judgment">Final judgment</button><button class="tab" data-tab="audit">Acceptance audit</button></nav><section id="tab-trajectory" class="tab-page active">${kickoffAgentsHtml(data.kickoff_agents_md)}${initialUserHtml(data.initial_user_message)}${turns.map((t,i)=>stepHtml(t,i===0)).join("")||'<p class="muted">No completed turns yet.</p>'}</section><section id="tab-judgment" class="tab-page">${judgmentHtml(j)}</section><section id="tab-audit" class="tab-page">${auditHtml(a)}</section>${danglingFinalHtml(s)}`;
  document.querySelectorAll(".tab").forEach(tab=>tab.onclick=()=>{document.querySelectorAll(".tab,.tab-page").forEach(x=>x.classList.remove("active"));tab.classList.add("active");$(`tab-${tab.dataset.tab}`).classList.add("active")})
}
function stepActions(t){return(Array.isArray(t.actions)&&t.actions.length)?t.actions:(t.action?[t.action]:[])}
function stepToolResults(t){return(Array.isArray(t.tool_results)&&t.tool_results.length)?t.tool_results:(t.tool_response!=null?[{tool_response:t.tool_response}]:[])}
function actionSummary(actions){if(!actions.length)return"action";const head=actions[0],label=`${head.name||head.kind||"action"} ${JSON.stringify(head.arguments||{})}`;return actions.length===1?label:`${label} · +${actions.length-1} sibling`}
function toolPayload(result){if(!result)return null;if(Object.prototype.hasOwnProperty.call(result,"tool_response"))return result.tool_response;return result}
function pairedToolBlocks(actions,results){const byId={};results.forEach(r=>{if(r&&r.action_id)byId[r.action_id]=r});const parts=[];actions.forEach((action,i)=>{const id=action&&action.id;const matched=(id&&byId[id])?byId[id]:results[i];const payload=toolPayload(matched);const actionTitle=actions.length>1?`Action ${i+1} / request`:"Action / request";const title=actions.length>1?`Tool ${i+1}: ${action.name||action.kind||"action"}`:"Tool response";parts.push(block(actionTitle,pretty(action),true));if(payload===null||payload===undefined){parts.push(`<div class="block"><h4>${esc(title)}</h4><div class="prose muted">No tool observation recorded for this sibling.</div></div>`)}else{parts.push(block(title,pretty(payload),true))}});results.slice(actions.length).forEach((r,i)=>{const payload=toolPayload(r);parts.push(block(`Extra tool result ${actions.length+i+1}`,pretty(payload),true))});return parts.join("")}
function stepHtml(t,openFirst=false){
  const actions=stepActions(t),results=stepToolResults(t),ra=t.reasoning_assessment||{},issues=t.integrity_issues||[],kind=issues.length?"bad":t.trainable===false?"warn":"ok";
  const terminalChip=t.agent_terminal_stop?pill("then agent terminal","warn"):(t.terminate_episode?pill("world terminal","ok"):"");
  const agentText=reasoningByType(t.reasoning,"text").join(" ").replace(/\s+/g," ").trim();
  const summaryLabel=agentText?`${agentText.slice(0,120)}${agentText.length>120?"…":""} · ${actionSummary(actions)}`:actionSummary(actions);
  const open=openFirst||t.agent_terminal_stop;
  // Pre-action agent speech first; tool JSON second. Collapsed accordion + tools-first
  // ordering made the initial Player response look missing.
  return`<details class="step ${t.agent_terminal_stop?"agent-terminal-step":""}" ${open?"open":""}><summary><span class="step-index">#${String(t.index).padStart(3,"0")}</span><span class="step-action">${esc(summaryLabel)}</span><span class="score-stack"><span class="score ${scoreClass(t.turn_reward)}">R ${fmt(t.turn_reward)}</span><span class="score ${scoreClass(ra.quality)}">Q ${fmt(ra.quality)}</span><span class="score ${scoreClass(ra.efficiency)}">E ${fmt(ra.efficiency)}</span></span>${terminalChip}${pill(issues.length?`${issues.length} issue`:"clean",kind)}${actions.length>1?pill(`${actions.length} tools`,"ok"):""}</summary><div class="step-body"><div class="detail-grid">${datum("Reward",fmt(t.turn_reward,3))}${datum("Quality",fmt(ra.quality))}${datum("Efficiency",fmt(ra.efficiency))}${datum("Completion",fmt(t.completion_percentage,0)+"%")}${datum("Tools",actions.length)}</div>${(t.reasoning||[]).map(r=>block(`Agent ${r.type}`,r.text)).join("")}${pairedToolBlocks(actions,results)}${block("Reasoning assessment",ra.commentary)}${block("World commentary",t.commentary)}${issues.length?block("Integrity issues",pretty(issues),true):""}${t.termination_reason?block("Termination",t.termination_reason):""}${t.agent_terminal_stop?`<div class="block dangling-inline"><h4>After this world reply</h4><div class="prose muted">Player produced a final assistant message with no tool call (see dangling final above). No further world turn was created.</div></div>`:""}</div></details>`
}
function block(title,content,code=false){if(content===null||content===undefined||content==="")return"";return`<div class="block"><h4>${esc(title)}</h4>${code?`<pre>${esc(content)}</pre>`:`<div class="prose">${esc(content)}</div>`}</div>`}
function judgmentHtml(j){if(!Object.keys(j).length)return'<p class="muted">Terminal judgment has not been produced.</p>';return`<div class="judgment"><h3>Terminal result · reward ${fmt(j.final_reward,3)}</h3><p>${esc(j.commentary)}</p><div class="badges">${j.trainable?pill("trainable","ok"):pill("untrainable","bad")}${(j.integrity_issues||[]).length?pill(`${j.integrity_issues.length} integrity issues`,"bad"):pill("integrity clean","ok")}</div></div><h3>Reward criteria</h3>${(j.criteria||[]).map(c=>`<div class="criterion"><strong>${esc(c.criterion)}</strong><span class="score ${scoreClass((c.score||0)/(c.weight||1))}">${fmt(c.score,3)} / ${fmt(c.weight,2)}</span><span>${esc(c.evidence)}</span></div>`).join("")}${block("Process deductions",pretty(j.process_deductions||[]),true)}${block("Task outcomes",pretty(j.task_outcomes),true)}${block("Factual consistency",pretty(j.factual_consistency),true)}`}
function auditHtml(a){if(!Object.keys(a).length)return'<p class="muted">Acceptance audit has not been produced.</p>';const sections=Object.entries(a).filter(([,v])=>v&&typeof v==="object");return`<div class="judgment"><h3>Acceptance · ${a.accepted_for_training===true?"PASS":a.accepted_for_training===false?"FAIL":"UNKNOWN"}</h3><div class="badges">${a.accepted_for_training===true?pill("accepted","ok"):pill("not accepted","bad")}</div></div>${sections.map(([name,value])=>block(name.replaceAll("_"," "),pretty(value),true)).join("")}`}
async function init(){try{const response=await fetch("/api/runs"),data=await response.json();if(data.error)showError(data.error);state.runs=data.runs||[];setOptions("campaignFilter",state.runs.map(r=>r.campaign).filter(Boolean));setOptions("runFilter",state.runs.map(r=>r.run_dir));setOptions("taskFilter",state.runs.map(r=>r.task));setOptions("modelFilter",state.runs.map(r=>r.player_model));["search","campaignFilter","runFilter","taskFilter","modelFilter","stateFilter","rewardFilter"].forEach(id=>$(id).addEventListener("input",applyFilters));$("groupBy").addEventListener("input",renderCatalog);applyFilters();if(state.runs.length)selectRun(state.runs[0].id)}catch(error){showError(`Could not load runs: ${error.message}`)}}
init();
