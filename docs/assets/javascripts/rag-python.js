// Talks to the Python worker (rag-worker.js), shared by every playground page.
//   const python = RagPython.start(statusElement);
//   const info = await python("info");             // one call into rag_titan.playground
// The worker starts on the first call, so pages without a playground load nothing.

const RagPython = (() => {
  // read while this script runs: later, document.currentScript is another script
  const here = document.currentScript.src;   // .../assets/javascripts/rag-python.js?v=...
  const assets = new URL("../", here).href;

  function start(status) {
    const worker = new Worker(new URL(`rag-worker.js${new URL(here).search}`, here),
                              { type: "module" });
    const waiting = {};
    let calls = 0;

    worker.onmessage = ({ data }) => {
      if (data.status) { status.textContent = data.status; return; }
      const { resolve, reject } = waiting[data.id];
      delete waiting[data.id];
      if (data.error) reject(new Error(data.error)); else resolve(data.result);
    };
    // a worker that fails to load (no network, old browser) must not leave "Starting…" forever
    worker.onerror = (event) => {
      status.textContent = "The playground could not start: "
        + `${event.message || "the Python worker failed to load"}`;
    };
    return (call, ...args) => new Promise((resolve, reject) => {
      waiting[++calls] = { resolve, reject };
      worker.postMessage({ id: calls, call, args, assets });
    });
  }

  return { start };
})();
