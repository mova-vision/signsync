/* ------------------------------------------------------------
   mv-dropdown — upgrades every admin <select> into a playground-
   style flat menu (the camera-picker look). The native select
   stays in the DOM as the hidden source of truth: it keeps its
   id / value / 'change' events, so all existing page JS works
   untouched. New selects (re-rendered rows) are picked up via
   MutationObserver.
   ------------------------------------------------------------ */
(function () {
  const CHEVRON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>';

  function upgrade(sel) {
    if (sel.dataset.upgraded) return;
    sel.dataset.upgraded = '1';

    const wrap = document.createElement('span');
    wrap.className = 'mv-dd-wrap';
    sel.parentNode.insertBefore(wrap, sel);
    wrap.append(sel);

    const val = document.createElement('span');
    val.className = 'mv-dd-val';
    const chev = document.createElement('span');
    chev.className = 'mv-dd-chev';
    chev.innerHTML = CHEVRON;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mv-dd-btn';
    btn.append(val, chev);

    const menu = document.createElement('span');
    menu.className = 'mv-dd-menu';

    const label = () => {
      const o = sel.options[sel.selectedIndex];
      return o ? o.textContent : '';
    };

    function renderMenu() {
      menu.replaceChildren(
        ...Array.from(sel.options).map((opt) => {
          const it = document.createElement('button');
          it.type = 'button';
          it.className = 'mv-dd-item' + (opt.selected ? ' is-active' : '');
          it.textContent = opt.textContent;
          if (opt.disabled) it.classList.add('is-disabled');
          it.onclick = () => {
            if (opt.disabled) return;
            if (sel.value !== opt.value) {
              sel.value = opt.value;
              sel.dispatchEvent(new Event('change', { bubbles: true }));
            }
            sync();
            close();
          };
          return it;
        })
      );
    }

    function sync() {
      val.textContent = label();
      val.title = label();
    }

    function open() {
      if (sel.disabled) return;
      document.querySelectorAll('.mv-dd-wrap.is-open').forEach((w) => {
        if (w !== wrap) w.classList.remove('is-open');
      });
      renderMenu();
      wrap.classList.add('is-open');
      const act = menu.querySelector('.is-active');
      if (act) act.scrollIntoView({ block: 'nearest' });
    }
    function close() {
      wrap.classList.remove('is-open');
    }

    btn.onclick = () => {
      if (sel.disabled) return;
      wrap.classList.contains('is-open') ? close() : open();
    };
    sel.addEventListener('change', sync);

    document.addEventListener('click', (e) => {
      if (!wrap.contains(e.target)) close();
    });

    wrap.append(btn, menu);
    sync();
  }

  const scan = () =>
    document.querySelectorAll('select:not([data-upgraded])').forEach(upgrade);

  const mo = new MutationObserver(() => scan());
  mo.observe(document.body, { childList: true, subtree: true });

  if (document.readyState !== 'loading') scan();
  else document.addEventListener('DOMContentLoaded', scan);
})();
