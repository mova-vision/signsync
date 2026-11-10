/* ------------------------------------------------------------
   mv-workspace — admin topbar workspace selector.
   The backend owns the state: this fetches /api/workspaces each
   load, marks the active one selected, and creating a new one is
   a prompt → POST → switch → reload. On server restart the
   backend boots on "default" (by design).
   ------------------------------------------------------------ */
(function () {
  function boot() {
    const host = document.getElementById('mvWorkspaces');
    if (!host) return;

    async function rebuild(data) {
      host.replaceChildren();
      const sel = document.createElement('select');
      sel.id = 'wsSelect';
      sel.title = 'Workspace — switches the entire dataset: gestures, groups, captures, audio';

      (data.workspaces || ['default']).forEach((name) => {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = name === 'default' ? 'workspace: default' : `workspace: ${name}`;
        if (name === data.active) o.selected = true;
        sel.append(o);
      });
      const o = document.createElement('option');
      o.value = '__new__';
      o.textContent = '+ new workspace…';
      sel.append(o);
      sel.dataset.active = data.active;
      host.append(sel); // add once options are in → mv-dropdown syncs label

      sel.onchange = async () => {
        const val = sel.value;
        if (val === '__new__') {
          const name = (prompt('New workspace name (a-z 0-9 - _, e.g. experiment1 or user2):') || '').trim().toLowerCase();
          if (!name) return rebuild(data); // cancelled → restore
          const cres = await fetch('/api/workspaces', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
          });
          if (!cres.ok) {
            const d = await cres.json().catch(() => ({}));
            alert(d.detail || 'could not create workspace');
            return rebuild(data);
          }
          await fetch('/api/workspaces/switch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
          });
          location.reload();
          return;
        }
        if (val === data.active) return; // no-op
        await fetch('/api/workspaces/switch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: val })
        });
        location.reload();
      };
    }

    fetch('/api/workspaces').then(r => r.json()).then(rebuild);
  }

  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
