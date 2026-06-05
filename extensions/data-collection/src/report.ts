/** Small helpers shared across commands: a report dialog and robust error formatting. */

import type { initialize } from "@ableton-extensions/sdk";

type Ctx = ReturnType<typeof initialize>;

export function formatError(e: unknown): string {
  if (e === undefined) return "undefined (a non-Error value was thrown)";
  if (e === null) return "null";
  if (e instanceof Error) return e.stack ?? `${e.name}: ${e.message}`;
  if (typeof e === "string") return e;
  try {
    return `[${typeof e}] ${JSON.stringify(e)}`;
  } catch {
    return `[${typeof e}] ${String(e)}`;
  }
}

export async function report(context: Ctx, lines: string[]): Promise<void> {
  const text = lines.join("\n");
  console.log("[doppelganger]\n" + text);

  const html = `<!doctype html><html><head><meta charset="utf-8">
<style>
  body { font: 13px/1.5 -apple-system, Segoe UI, sans-serif; margin: 16px; background:#1e1e1e; color:#eee; }
  pre  { white-space: pre-wrap; word-break: break-word; }
  button { margin-top: 12px; padding: 6px 18px; font-size: 13px; }
</style></head>
<body>
  <h3>doppelganger — Data Collection</h3>
  <pre>${escapeHtml(text)}</pre>
  <button onclick="done()">OK</button>
  <script>
    function done() {
      if (window.chrome && window.chrome.webview) window.chrome.webview.postMessage("ok");
      else if (window.webkit) window.webkit.messageHandlers.live.postMessage("ok");
    }
  </script>
</body></html>`;

  await context.ui.showModalDialog(
    `data:text/html,${encodeURIComponent(html)}`,
    560,
    460,
  );
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
