import { useEffect, useRef } from "react";
import type { LiveAgent, LiveSlot } from "../lib/api";

function pct(n: number, d: number): number {
  return d > 0 ? Math.min(100, Math.round((100 * n) / d)) : 0;
}

/** One-line live status for the agent list. */
export function LiveBadge({ live }: { live?: LiveAgent }) {
  if (!live) return null;
  const slot = live.slot;
  let text: string;
  if (live.state === "waiting") {
    text = slot && slot.to_read > 0 ? `reading ${pct(slot.read, slot.to_read)}%` : "waiting for model";
  } else if (live.state === "streaming") {
    text = slot ? `generating · ${slot.decoded.toLocaleString()} tok` : "generating";
  } else if (live.state === "tool") {
    text = `running ${live.tool || "tool"}`;
  } else if (live.state === "between") {
    text = "sending next request";
  } else {
    text = "idle";
  }
  const active = live.state === "waiting" || live.state === "streaming";
  return (
    <span className="flex items-center gap-1.5">
      <span className={`w-1.5 h-1.5 rounded-full ${active ? "bg-green-500 animate-pulse" : "bg-border-strong"}`} />
      {text}
    </span>
  );
}

function Bar({ label, value, total, detail }: { label: string; value: number; total: number; detail: string }) {
  return (
    <div>
      <div className="flex justify-between font-mono text-[11px] text-muted-fg mb-1">
        <span>{label}</span>
        <span>{detail}</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
        <div className="h-full bg-foreground transition-[width] duration-500" style={{ width: `${pct(value, total)}%` }} />
      </div>
    </div>
  );
}

function SlotProgress({ slot }: { slot: LiveSlot }) {
  // Once tokens are being generated the prompt has been fully read, even if
  // llama.cpp's read counter stops short of the total.
  const generating = slot.decoded > 0;
  const read = generating ? slot.to_read : slot.read;
  const reading = !generating;
  return (
    <div className="space-y-2">
      <Bar
        label={`prompt · slot ${slot.id} (estimated match)`}
        value={slot.cached + read}
        total={slot.prompt}
        detail={`${(slot.cached + read).toLocaleString()} / ${slot.prompt.toLocaleString()} tokens · ${slot.cached.toLocaleString()} from cache, ${read.toLocaleString()} / ${slot.to_read.toLocaleString()} read`}
      />
      <p className="font-mono text-[11px] text-muted-fg">
        {reading ? "reading prompt…" : `generated ${slot.decoded.toLocaleString()} tokens`}
      </p>
    </div>
  );
}

/** Live view of the selected agent's in-flight model call. */
export default function LivePanel({ live, showText = true }: { live?: LiveAgent; showText?: boolean }) {
  const streamRef = useRef<HTMLPreElement>(null);
  const text = (live?.entries ?? [])
    .map((e) => (e.type === "tool_call" ? `→ ${e.content}: ${e.details?.input ?? ""}` : e.content))
    .join("\n\n");

  useEffect(() => {
    const el = streamRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [text]);

  if (!live || live.state === "idle") return null;

  const heading =
    live.state === "waiting" ? "Waiting for the model (reading prompt)"
    : live.state === "streaming" ? "Model is responding"
    : live.state === "tool" ? `Running tool: ${live.tool || "?"}`
    : "Between model calls";

  return (
    <div className="mb-4 p-3 rounded-lg border border-border bg-muted/40 space-y-3">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tracking-widest uppercase text-muted-fg">Live · {heading}</span>
        {live.context_tokens ? (
          <span className="font-mono text-[11px] text-muted-fg">
            last context {live.context_tokens.toLocaleString()} tokens
          </span>
        ) : null}
      </div>
      {live.slot && <SlotProgress slot={live.slot} />}
      {showText && live.state === "streaming" && text && (
        <pre
          ref={streamRef}
          className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto text-muted-fg"
        >
          {text}
        </pre>
      )}
    </div>
  );
}
