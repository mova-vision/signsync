/* ------------------------------------------------------------
   4c. Card 4 · AI Processing panel
   Every number here arrives measured from the pipeline stages —
   nothing is hardcoded. When the MiniLM stage is switched off in
   .env (AI_MINILM_ON=0) its row reports "off" and the decision
   latency keeps flowing untouched.
   ------------------------------------------------------------ */
(function () {
  const panel = $('pgAiPanel');
  if (!panel) return;

  const status = $('pgAiStatus');
  const tagsVal = $('pgAiTags');
  const permVal = $('pgAiPerm');
  const minimlVal = $('pgAiMiniml');
  const minimlName = $('pgAiMinimlName');
  const decisionVal = $('pgAiDecision');
  const decisionName = $('pgAiDecisionName');
  const sentence = $('pgAiSentence');
  const stageEls = {
    tags: $('pgAiStageTags'),
    perm: $('pgAiStagePerm'),
    miniml: $('pgAiStageMiniml'),
    decision: $('pgAiStageDecision'),
  };

  function setLoading(on, label) {
    panel.classList.toggle('is-loading', !!on);
    status.classList.toggle('is-loading', !!on);
    status.classList.toggle('is-idle', !on && !(label && label !== 'Idle'));
    status.classList.remove('is-error');
    if (!on) {
      status.textContent = label || 'Idle';
      Object.values(stageEls).forEach(li => li.classList.remove('is-active'));
    } else {
      status.textContent = label || 'Thinking…';
    }
  }

  function fmt(ms) {
    if (ms == null || ms === undefined) return '—';
    return `${Number(ms).toFixed(1)} ms`;
  }

  function renderTags(state, display) {
    const tags = (state && state.tags) || [];
    if (tags.length) {
      tagsVal.textContent = String(tags.length);
      tagsVal.title = tags.map(t => String(t).replace(/_/g, ' ')).join(' → ');
      stageEls.tags.classList.toggle(display, true);
      return;
    }
    tagsVal.textContent = display === 'is-done' ? '0' : '—';
  }

  function renderPending(state) {
    if (!state) return;
    if (state.result !== 'pending') return renderFinished(state);
    setLoading(true, 'Thinking…');
    renderTags(state, 'is-active');
    permVal.textContent = '…';
    minimlVal.textContent = state.miniml_enabled === false ? 'off' : '…';
    decisionVal.textContent = state.decision_available === false ? 'no .env creds' : '…';
    sentence.textContent = 'AI pipeline running — permutations → MiniLM → clef…';
    sentence.classList.remove('is-error');
  }

  function renderFinished(ev) {
    if (!ev || ev.result === 'pending') return;
    setLoading(false, ev.result === 'ok' ? 'Done' : 'Error');

    renderTags(ev, 'is-done');

    permVal.textContent = `${ev.option_count != null ? ev.option_count : '—'} options · ${fmt(ev.perm_ms)}`;
    stageEls.perm.classList.add('is-done');

    minimlVal.classList.remove('is-off');
    if (ev.miniml_enabled === false) {
      minimlVal.textContent = 'off';
      minimlVal.classList.add('is-off');
    } else if (!ev.miniml_ms) {
      minimlVal.textContent = 'pass-through · 0.0 ms';
    } else {
      minimlVal.textContent = `${ev.option_count_after != null ? ev.option_count_after : '—'} kept · ${fmt(ev.miniml_ms)}`;
    }
    minimlVal.title = 'MiniLM (ms-marco-MiniLM-L-6-v2) reduces candidates to top-K when the set is big';
    stageEls.miniml.classList.add('is-done');
    if (ev.miniml_enabled === false) stageEls.miniml.classList.remove('is-done');

    if (ev.result === 'ok') {
      decisionVal.textContent = ev.decision_note
        ? `auto-picked · 0.0 ms`
        : `${ev.decision_ms != null ? fmt(ev.decision_ms) : '—'}${ev.confidence != null ? ` · conf ${Number(ev.confidence).toFixed(2)}` : ''}`;
      decisionVal.title = ev.decision_note || 'Cloudflare clef-flash decision call latency';
      decisionVal.classList.remove('is-error');
      decisionVal.classList.add('is-done');
      stageEls.decision.classList.remove('is-error');
      stageEls.decision.classList.add('is-done');
      sentence.textContent = ev.sentence || '(no sentence)';
      sentence.classList.remove('is-error');
      status.textContent = ev.total_ms != null ? `Done · ${fmt(ev.total_ms)} total` : 'Done';
    } else {
      decisionVal.textContent = 'failed';
      decisionVal.title = ev.error || '';
      decisionVal.classList.add('is-error');
      decisionVal.classList.remove('is-done');
      stageEls.decision.classList.add('is-error');
      stageEls.decision.classList.remove('is-done');
      sentence.textContent = ev.error || 'Decision failed';
      sentence.classList.add('is-error');
      status.textContent = 'Error';
      status.classList.add('is-error');
    }
  }

  pipelineListeners.push((data) => {
    if (!data) return;
    if (data.ai_event) renderFinished(data.ai_event);
    else if (data.ai_state) renderPending(data.ai_state);
  });
})();
