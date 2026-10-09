// Routes the screens' requests to the Rust engine.
//
// The UI was written against the old Python engine's local web server
// (fetch('/notes'), fetch('/settings'), …). There is no server any more:
// this replaces window.fetch for app paths with a call into the Rust engine
// and hands back a normal Response, so the screens work unchanged — same
// paths, same JSON, same {"detail": …} errors. Other URLs use the real fetch.
(() => {
  const { invoke } = window.__TAURI__.core;
  const realFetch = window.fetch.bind(window);

  // Script errors, for the development self-check (main.rs debug_report).
  window.__itsmypaErrors = [];
  window.addEventListener('error', e => window.__itsmypaErrors.push(String(e.message)));
  window.addEventListener('unhandledrejection', e => window.__itsmypaErrors.push(String(e.reason)));

  // Request body → plain JSON. Form fields become strings. An attached
  // recording ("audio") is sent to the engine as raw bytes and replaced by its
  // {upload_id}; other attached files become {name, text} (backups are JSON).
  async function toJson(body) {
    if (body == null) return null;
    if (typeof body === 'string') {
      try { return JSON.parse(body); } catch { return body; }
    }
    if (body instanceof FormData) {
      const out = {};
      for (const [key, value] of body.entries()) {
        if (key === 'audio' && value instanceof Blob) {
          const bytes = new Uint8Array(await value.arrayBuffer());
          const name = value.name || 'audio';
          out[key] = { upload_id: await invoke('upload', bytes, { headers: { 'x-file-name': name } }) };
        } else {
          out[key] = value instanceof File ? { name: value.name, text: await value.text() } : value;
        }
      }
      return out;
    }
    return body;
  }

  const jsonResponse = (data, status = 200) =>
    new Response(JSON.stringify(data ?? null), { status, headers: { 'Content-Type': 'application/json' } });

  // AI answers stream in as they're written (summary, chat, …).
  const STREAMING = new Set(['/summarize', '/enhance', '/generate/email', '/chat', '/ask']);

  // Resolves with a streaming Response once the first piece arrives (or the
  // answer ends) — so a bad request still comes back as a proper error status.
  function streamResponse(path, body) {
    return new Promise(resolve => {
      const enc = new TextEncoder();
      let controller, started = false;
      const stream = new ReadableStream({ start(c) { controller = c; } });
      const begin = () => { if (!started) { started = true; resolve(new Response(stream, { status: 200 })); } };
      const onPiece = new window.__TAURI__.core.Channel();
      onPiece.onmessage = piece => { begin(); controller.enqueue(enc.encode(piece)); };
      invoke('api_stream', { path, body, onPiece }).then(
        () => { begin(); controller.close(); },
        err => {
          if (started) { controller.close(); return; }
          started = true;
          const e = err && typeof err === 'object' ? err : { status: 500, detail: String(err) };
          resolve(jsonResponse({ detail: e.detail || 'Something went wrong' }, e.status || 500));
        });
    });
  }

  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (!url.startsWith('/')) return realFetch(input, init);
    const path = url.split('?')[0];
    const method = (init.method || 'GET').toUpperCase();
    if (method === 'POST' && STREAMING.has(path)) return streamResponse(path, await toJson(init.body));
    try {
      const data = await invoke('api', { method, path, body: await toJson(init.body) });
      return jsonResponse(data);
    } catch (err) {
      const e = err && typeof err === 'object' ? err : { status: 500, detail: String(err) };
      return jsonResponse({ detail: e.detail || 'Something went wrong' }, e.status || 500);
    }
  };
})();
