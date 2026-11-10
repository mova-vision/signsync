/* ------------------------------------------------------------
   4c. Card 4 · AI Processing panel
   Every number here arrives measured from the pipeline stages —
   nothing is hardcoded. Chain: ordered readings → one LLM call → TTS.
   ------------------------------------------------------------ */
(function () {
  const panel = $('pgAiPanel');
  if (!panel) return;

  const status = $('pgAiStatus');
  const tagsVal = $('pgAiTags');
  const permVal = $('pgAiPerm');
  const llmVal = $('pgAiDecision');
  const llmName = $('pgAiDecisionName');
  const sentence = $('pgAiSentence');
  const stageEls = {
    tags: $('pgAiStageTags'),
    perm: $('pgAiStagePerm'),
    llm: $('pgAiStageDecision'),
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
    llmVal.textContent = state.llm_available === false ? 'no .env creds' : '…';
    sentence.textContent = 'AI pipeline running — gestures → LLM…';
    sentence.classList.remove('is-error');
  }

  function renderFinished(ev) {
    if (!ev || ev.result === 'pending') return;
    setLoading(false, ev.result === 'ok' ? 'Done' : 'Error');

    renderTags(ev, 'is-done');

    permVal.textContent = `${ev.option_count != null ? ev.option_count : '—'} readings · ${fmt(ev.perm_ms)}`;
    stageEls.perm.classList.add('is-done');
    permVal.title = 'Possible meaning combinations across the ordered gestures';

    if (ev.model) llmName.textContent = `LLM · ${ev.model}`;

    if (ev.result === 'ok') {
      llmVal.textContent = ev.decision_note
        ? 'auto-picked · 0.0 ms'
        : `${ev.llm_ms != null ? fmt(ev.llm_ms) : '—'}`;
      llmVal.title = ev.decision_note || 'OpenAI-compatible chat completion latency';
      llmVal.classList.remove('is-error');
      llmVal.classList.add('is-done');
      stageEls.llm.classList.remove('is-error');
      stageEls.llm.classList.add('is-done');
      sentence.textContent = ev.sentence || '(no sentence)';
      sentence.classList.remove('is-error');
      status.textContent = ev.total_ms != null ? `Done · ${fmt(ev.total_ms)} total` : 'Done';
    } else {
      llmVal.textContent = 'failed';
      llmVal.title = ev.error || '';
      llmVal.classList.add('is-error');
      llmVal.classList.remove('is-done');
      stageEls.llm.classList.add('is-error');
      stageEls.llm.classList.remove('is-done');
      sentence.textContent = ev.error || 'LLM call failed';
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
