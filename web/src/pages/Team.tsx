import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
import "chartjs-adapter-date-fns";
import { Line } from "react-chartjs-2";
import {
  api,
  type InteractionKind,
  type MemoryKind,
  type TeamInteraction,
  type InferenceSample,
  type TeamResponse,
} from "../lib/api";
import { useSSE } from "../hooks/useSSE";
import { useInference } from "../hooks/useInference";
import { useTheme, type Theme } from "../hooks/useTheme";
import { agentColor } from "../lib/agentColors";

ChartJS.register(LinearScale, TimeScale, PointElement, LineElement, Tooltip, Legend);

/* Interaction kinds: colour per theme, dash pattern, curve offset (so parallel
   links between the same pair don't overlap), and legend text. */
const KINDS: Record<
  InteractionKind,
  { light: string; dark: string; dash?: string; bend: number; label: string }
> = {
  message: { light: "#1f6feb", dark: "#6cb6ff", bend: 0.28, label: "message" },
  broadcast: { light: "#8250df", dark: "#c297ff", bend: 0.14, label: "broadcast" },
  built_on: { light: "#1a7f37", dark: "#57ab5a", dash: "7 5", bend: -0.14, label: "built on code" },
  note_ref: { light: "#9a6700", dark: "#d4a72c", dash: "2 5", bend: -0.28, label: "cited note" },
};

const MEMORY: Record<MemoryKind, { light: string; dark: string; label: string }> = {
  attempt: { light: "#374151", dark: "#b4bcc9", label: "attempts" },
  note: { light: "#9a6700", dark: "#d4a72c", label: "notes" },
  skill: { light: "#1a7f37", dark: "#57ab5a", label: "skills" },
  message: { light: "#1f6feb", dark: "#6cb6ff", label: "messages" },
};

const VW = 640;
const VH = 460;
const REPLAY_SECONDS = 40; // a full replay takes this long at 1x

type Selection = { agent: string } | { from: string; to: string; kind: InteractionKind } | null;

function clock(t: number): string {
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function elapsed(t: number, t0: number): string {
  const m = Math.max(0, Math.round((t - t0) / 60));
  return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}`;
}

function matches(i: TeamInteraction, sel: Selection): boolean {
  if (!sel) return true;
  if ("agent" in sel) return i.from === sel.agent || i.to === sel.agent;
  return i.from === sel.from && i.to === sel.to && i.kind === sel.kind;
}

export default function Team() {
  const { theme } = useTheme();
  const [data, setData] = useState<TeamResponse | null>(null);
  const [cursor, setCursor] = useState<number | null>(null); // null = live (show everything)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selection, setSelection] = useState<Selection>(null);
  const [feed, setFeed] = useState<"messages" | "all">("messages");

  const inference = useInference(5000);
  const refresh = () => api.team().then(setData).catch(() => {});
  useEffect(() => {
    refresh();
  }, []);
  useSSE({
    "attempt:new": refresh,
    "attempt:update": refresh,
    "note:update": refresh,
    "message:update": refresh,
  });

  const t0 = data?.run_start ?? data?.memory[0]?.t ?? 0;
  const t1 = Math.max(data?.now ?? 0, data?.memory.at(-1)?.t ?? 0, t0 + 1);
  const at = cursor ?? t1;

  // Replay: advance the cursor across the run in REPLAY_SECONDS / speed.
  const lastTick = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) {
      lastTick.current = null;
      return;
    }
    let frame = 0;
    const step = (now: number) => {
      const dt = lastTick.current == null ? 0 : (now - lastTick.current) / 1000;
      lastTick.current = now;
      setCursor((c) => {
        const next = (c ?? t0) + (dt * speed * (t1 - t0)) / REPLAY_SECONDS;
        if (next >= t1) {
          setPlaying(false);
          return null;
        }
        return next;
      });
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [playing, speed, t0, t1]);

  const visible = useMemo(() => {
    if (!data) return null;
    return {
      interactions: data.interactions.filter((i) => i.t <= at),
      memory: data.memory.filter((m) => m.t <= at),
      messages: data.messages.filter((m) => m.t <= at),
    };
  }, [data, at]);

  if (!data || !visible) {
    return <p className="col-span-2 p-8 font-mono text-xs text-muted-fg">Loading team…</p>;
  }

  return (
    <>
      <div className="flex flex-col min-h-0 border-r border-border">
        <TeamGraph
          agents={data.agents}
          visible={visible}
          theme={theme}
          selection={selection}
          onSelect={setSelection}
          recentAfter={at - (t1 - t0) * 0.03}
        />
        <Scrubber
          t0={t0}
          t1={t1}
          at={at}
          live={cursor == null}
          playing={playing}
          speed={speed}
          onSeek={(t) => setCursor(t >= t1 ? null : t)}
          onPlay={() => {
            if (cursor == null) setCursor(t0);
            setPlaying((p) => !p);
          }}
          onLive={() => {
            setPlaying(false);
            setCursor(null);
          }}
          onSpeed={setSpeed}
        />
        <MemoryChart memory={visible.memory} t0={t0} t1={t1} theme={theme} />
        {inference && inference.samples.length > 0 && (
          <ThroughputChart
            samples={inference.samples.filter((x) => x.t <= at)}
            t0={t0}
            t1={t1}
            theme={theme}
          />
        )}
      </div>
      <Feed
        data={data}
        visible={visible}
        theme={theme}
        t0={t0}
        feed={feed}
        onFeed={setFeed}
        selection={selection}
        onClear={() => setSelection(null)}
      />
    </>
  );
}

/* ---------------------------------------------------------------- graph -- */

function TeamGraph({
  agents,
  visible,
  theme,
  selection,
  onSelect,
  recentAfter,
}: {
  agents: string[];
  visible: { interactions: TeamInteraction[]; memory: TeamResponse["memory"] };
  theme: Theme;
  selection: Selection;
  onSelect: (s: Selection) => void;
  recentAfter: number;
}) {
  const pos = useMemo(() => {
    const r = Math.min(VW, VH) * 0.34;
    const out: Record<string, { x: number; y: number }> = {};
    agents.forEach((a, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(agents.length, 1);
      out[a] = { x: VW / 2 + r * Math.cos(angle), y: VH / 2 + r * Math.sin(angle) };
    });
    return out;
  }, [agents]);

  const stats = useMemo(() => {
    const s: Record<string, { attempts: number; best: number | null; recent: boolean }> = {};
    for (const a of agents) s[a] = { attempts: 0, best: null, recent: false };
    for (const m of visible.memory) {
      if (!m.agent || !s[m.agent]) continue;
      if (m.t > recentAfter) s[m.agent].recent = true;
      if (m.kind !== "attempt") continue;
      s[m.agent].attempts += 1;
      if (m.score != null && (s[m.agent].best == null || m.score > s[m.agent].best!)) {
        s[m.agent].best = m.score;
      }
    }
    return s;
  }, [agents, visible.memory, recentAfter]);

  const links = useMemo(() => {
    const agg = new Map<string, { from: string; to: string; kind: InteractionKind; count: number; recent: boolean }>();
    for (const i of visible.interactions) {
      if (!pos[i.from] || !pos[i.to]) continue;
      const key = `${i.from}|${i.to}|${i.kind}`;
      const cur = agg.get(key) ?? { from: i.from, to: i.to, kind: i.kind, count: 0, recent: false };
      cur.count += 1;
      cur.recent ||= i.t > recentAfter;
      agg.set(key, cur);
    }
    return [...agg.values()];
  }, [visible.interactions, pos, recentAfter]);

  const nodeR = (a: string) => 16 + 3.5 * Math.sqrt(stats[a]?.attempts ?? 0);
  const dim = (match: boolean) => (selection && !match ? 0.15 : 1);

  return (
    <div className="flex-1 min-h-0 relative">
      <div className="absolute top-3 left-4 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-muted-fg z-10">
        {(Object.keys(KINDS) as InteractionKind[]).map((k) => (
          <span key={k} className="flex items-center gap-1.5">
            <svg width="22" height="6">
              <line
                x1="0" y1="3" x2="22" y2="3"
                stroke={KINDS[k][theme]} strokeWidth="2" strokeDasharray={KINDS[k].dash}
              />
            </svg>
            {KINDS[k].label}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full h-full" onClick={() => onSelect(null)}>
        <defs>
          {(Object.keys(KINDS) as InteractionKind[]).map((k) => (
            <marker
              key={k} id={`team-arw-${k}`} viewBox="0 0 10 10" refX="9" refY="5"
              markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill={KINDS[k][theme]} />
            </marker>
          ))}
        </defs>

        {links.map((l) => {
          const a = pos[l.from], b = pos[l.to];
          const dx = b.x - a.x, dy = b.y - a.y;
          const len = Math.hypot(dx, dy) || 1;
          const bend = KINDS[l.kind].bend * len;
          const cx = (a.x + b.x) / 2 - (dy / len) * bend;
          const cy = (a.y + b.y) / 2 + (dx / len) * bend;
          // Stop the arrow at the target node's edge.
          const ex = b.x - cx, ey = b.y - cy, el = Math.hypot(ex, ey) || 1;
          const tx = b.x - (ex / el) * (nodeR(l.to) + 3);
          const ty = b.y - (ey / el) * (nodeR(l.to) + 3);
          const sel = matches({ ...l, t: 0, text: "" }, selection);
          return (
            <g
              key={`${l.from}|${l.to}|${l.kind}`}
              className="cursor-pointer"
              opacity={dim(sel)}
              onClick={(e) => {
                e.stopPropagation();
                onSelect({ from: l.from, to: l.to, kind: l.kind });
              }}
            >
              <path d={`M${a.x},${a.y} Q${cx},${cy} ${tx},${ty}`} stroke="transparent" strokeWidth="14" fill="none" />
              <path
                d={`M${a.x},${a.y} Q${cx},${cy} ${tx},${ty}`}
                fill="none"
                stroke={KINDS[l.kind][theme]}
                strokeWidth={1.2 + 1.6 * Math.sqrt(l.count)}
                strokeDasharray={KINDS[l.kind].dash}
                strokeOpacity={l.recent ? 1 : 0.6}
                markerEnd={`url(#team-arw-${l.kind})`}
              >
                <title>{`${l.from} → ${l.to}: ${l.count} × ${KINDS[l.kind].label}`}</title>
              </path>
              {l.count > 1 && (
                <text
                  x={cx} y={cy} textAnchor="middle" dominantBaseline="middle"
                  className="font-mono" fontSize="10" fill={KINDS[l.kind][theme]}
                >
                  {l.count}
                </text>
              )}
            </g>
          );
        })}

        {agents.map((a, i) => {
          const p = pos[a];
          const s = stats[a];
          const sel = !selection || ("agent" in selection ? selection.agent === a : selection.from === a || selection.to === a);
          return (
            <g
              key={a}
              className="cursor-pointer"
              opacity={dim(sel)}
              onClick={(e) => {
                e.stopPropagation();
                onSelect({ agent: a });
              }}
            >
              {s?.recent && (
                <circle cx={p.x} cy={p.y} r={nodeR(a) + 6} fill="none" stroke={agentColor(i, theme)} strokeOpacity="0.35" strokeWidth="3" />
              )}
              <circle cx={p.x} cy={p.y} r={nodeR(a)} fill={agentColor(i, theme)} />
              <text x={p.x} y={p.y + 4} textAnchor="middle" className="font-mono" fontSize="11" fill={theme === "dark" ? "#0e1011" : "#eff0f0"}>
                {s?.attempts ?? 0}
              </text>
              <text x={p.x} y={p.y + nodeR(a) + 15} textAnchor="middle" className="font-body fill-foreground" fontSize="12" stroke={theme === "dark" ? "#0e1011" : "#eff0f0"} strokeWidth="4" paintOrder="stroke">
                {a}
              </text>
              <text x={p.x} y={p.y + nodeR(a) + 29} textAnchor="middle" className="font-mono fill-muted-fg" fontSize="10" stroke={theme === "dark" ? "#0e1011" : "#eff0f0"} strokeWidth="4" paintOrder="stroke">
                {s?.best != null ? `best ${s.best.toFixed(1)}` : "no score yet"}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------- scrubber -- */

function Scrubber(props: {
  t0: number;
  t1: number;
  at: number;
  live: boolean;
  playing: boolean;
  speed: number;
  onSeek: (t: number) => void;
  onPlay: () => void;
  onLive: () => void;
  onSpeed: (s: number) => void;
}) {
  const { t0, t1, at, live, playing, speed } = props;
  return (
    <div className="flex items-center gap-3 px-4 py-2 border-t border-border font-mono text-[11px] text-muted-fg">
      <button onClick={props.onPlay} className="w-7 h-7 rounded-md hover:bg-muted text-foreground" title={playing ? "Pause" : "Replay"}>
        {playing ? "❚❚" : "▶"}
      </button>
      <input
        type="range"
        min={t0}
        max={t1}
        step={(t1 - t0) / 1000}
        value={at}
        onChange={(e) => props.onSeek(Number(e.target.value))}
        className="flex-1 accent-current"
      />
      <span className="w-24 text-right">{elapsed(at, t0)} · {clock(at)}</span>
      <select
        value={speed}
        onChange={(e) => props.onSpeed(Number(e.target.value))}
        className="bg-transparent border border-border rounded px-1 py-0.5"
      >
        {[0.5, 1, 2, 4].map((s) => (
          <option key={s} value={s}>{s}×</option>
        ))}
      </select>
      <button
        onClick={props.onLive}
        className={`px-2 py-0.5 rounded border ${live ? "border-green-500 text-green-600 dark:text-green-400" : "border-border hover:bg-muted"}`}
      >
        live
      </button>
    </div>
  );
}

/* --------------------------------------------------------- memory chart -- */

function MemoryChart({
  memory,
  t0,
  t1,
  theme,
}: {
  memory: TeamResponse["memory"];
  t0: number;
  t1: number;
  theme: Theme;
}) {
  const muted = theme === "dark" ? "#a6acae" : "#727879";
  const grid = theme === "dark" ? "#292d2f" : "#d0d3d5";
  const kinds = Object.keys(MEMORY) as MemoryKind[];

  const datasets = kinds.map((k) => {
    let n = 0;
    const pts = [{ x: t0 * 1000, y: 0 }];
    for (const m of memory) {
      if (m.kind !== k) continue;
      n += 1;
      pts.push({ x: m.t * 1000, y: n });
    }
    pts.push({ x: (memory.at(-1)?.t ?? t0) * 1000, y: n });
    return {
      label: `${MEMORY[k].label} (${n})`,
      data: pts,
      borderColor: MEMORY[k][theme],
      backgroundColor: MEMORY[k][theme],
      stepped: true as const,
      pointRadius: 0,
      borderWidth: 2,
      yAxisID: "y",
    };
  });

  let best: number | null = null;
  const bestPts: { x: number; y: number }[] = [];
  for (const m of memory) {
    if (m.kind === "attempt" && m.score != null && (best == null || m.score > best)) {
      best = m.score;
      bestPts.push({ x: m.t * 1000, y: best });
    }
  }
  if (best != null) bestPts.push({ x: (memory.at(-1)?.t ?? t0) * 1000, y: best });

  const tick = { font: { family: "'JetBrains Mono', monospace", size: 10 }, color: muted };
  return (
    <div className="h-56 shrink-0 border-t border-border px-4 pt-2 pb-3">
      <p className="font-mono text-[10px] tracking-widest uppercase text-muted-fg mb-1">Shared memory over time</p>
      <div className="h-[calc(100%-1.25rem)]">
        <Line
          data={{
            datasets: [
              ...datasets,
              {
                label: best != null ? `best score (${best.toFixed(1)})` : "best score",
                data: bestPts,
                borderColor: theme === "dark" ? "#eceeec" : "#1a1d1e",
                backgroundColor: theme === "dark" ? "#eceeec" : "#1a1d1e",
                borderDash: [4, 3],
                stepped: true as const,
                pointRadius: 0,
                borderWidth: 1.5,
                yAxisID: "y1",
              },
            ],
          }}
          options={{
            animation: false,
            maintainAspectRatio: false,
            parsing: false,
            interaction: { mode: "nearest", intersect: false },
            scales: {
              x: {
                type: "time",
                min: t0 * 1000,
                max: t1 * 1000,
                time: { displayFormats: { minute: "HH:mm", hour: "HH:mm" } },
                grid: { color: grid },
                ticks: tick,
              },
              y: { beginAtZero: true, grid: { color: grid }, ticks: { ...tick, precision: 0 } },
              y1: { position: "right", grid: { display: false }, ticks: tick },
            },
            plugins: {
              legend: { position: "top", align: "end", labels: { boxWidth: 10, font: tick.font, color: muted } },
            },
          }}
        />
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- throughput -- */

function ThroughputChart({
  samples,
  t0,
  t1,
  theme,
}: {
  samples: InferenceSample[];
  t0: number;
  t1: number;
  theme: Theme;
}) {
  const muted = theme === "dark" ? "#a6acae" : "#727879";
  const grid = theme === "dark" ? "#292d2f" : "#d0d3d5";
  const line = theme === "dark" ? "#eceeec" : "#1a1d1e";
  const tick = { font: { family: "'JetBrains Mono', monospace", size: 10 }, color: muted };
  const prompt = theme === "dark" ? "#d4a72c" : "#9a6700";
  const recent = samples.slice(-30); // ~1 minute at the 2 s sampling rate
  const mean = (f: (x: InferenceSample) => number) =>
    recent.length ? recent.reduce((a, x) => a + f(x), 0) / recent.length : 0;
  const genAvg = mean((x) => x.tok_s);
  const promptAvg = mean((x) => x.prompt_tok_s ?? 0);
  const total = samples.reduce((a, x, i) => (i ? a + x.tok_s * (x.t - samples[i - 1].t) : a), 0);

  return (
    <div className="h-36 shrink-0 border-t border-border px-4 pt-2 pb-3">
      <div className="flex items-baseline justify-between mb-1">
        <p className="font-mono text-[10px] tracking-widest uppercase text-muted-fg">Generation throughput</p>
        <p className="font-mono text-[11px] text-muted-fg">
          1 min avg: <span style={{ color: line }}>{Math.round(genAvg)} generating</span> ·{" "}
          <span style={{ color: prompt }}>{Math.round(promptAvg)} reading prompts</span> tok/s ·{" "}
          {(total / 1000).toFixed(0)}k generated
        </p>
      </div>
      <div className="h-[calc(100%-1.25rem)]">
        <Line
          data={{
            datasets: [
              {
                label: "generating tok/s",
                data: samples.map((x) => ({ x: x.t * 1000, y: x.tok_s })),
                borderColor: line,
                backgroundColor: line,
                pointRadius: 0,
                borderWidth: 1.2,
                yAxisID: "y",
              },
              {
                label: "reading prompts tok/s",
                data: samples.map((x) => ({ x: x.t * 1000, y: x.prompt_tok_s ?? 0 })),
                borderColor: prompt,
                backgroundColor: prompt,
                pointRadius: 0,
                borderWidth: 1,
                yAxisID: "y1",
              },
            ],
          }}
          options={{
            animation: false,
            maintainAspectRatio: false,
            parsing: false,
            scales: {
              x: {
                type: "time",
                min: t0 * 1000,
                max: t1 * 1000,
                time: { displayFormats: { minute: "HH:mm", hour: "HH:mm" } },
                grid: { color: grid },
                ticks: tick,
              },
              y: { beginAtZero: true, grid: { color: grid }, ticks: { ...tick, color: line } },
              y1: {
                beginAtZero: true,
                position: "right",
                grid: { display: false },
                ticks: { ...tick, color: prompt },
              },
            },
            plugins: { legend: { display: false } },
          }}
        />
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- feed -- */

function Feed({
  data,
  visible,
  theme,
  t0,
  feed,
  onFeed,
  selection,
  onClear,
}: {
  data: TeamResponse;
  visible: { interactions: TeamInteraction[]; memory: TeamResponse["memory"]; messages: TeamResponse["messages"] };
  theme: Theme;
  t0: number;
  feed: "messages" | "all";
  onFeed: (f: "messages" | "all") => void;
  selection: Selection;
  onClear: () => void;
}) {
  const items = useMemo(() => {
    const src =
      feed === "messages"
        ? visible.interactions.filter((i) => i.kind === "message" || i.kind === "broadcast")
        : visible.interactions;
    // A broadcast fans out to every agent; show it once in the feed.
    const seen = new Set<string>();
    return src
      .filter((i) => matches(i, selection))
      .filter((i) => {
        if (i.kind !== "broadcast") return true;
        const key = `${i.t}|${i.from}|${i.text}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .slice()
      .reverse();
  }, [visible.interactions, feed, selection]);

  const counts = (k: InteractionKind) => visible.interactions.filter((i) => i.kind === k).length;

  return (
    <div className="flex flex-col min-h-0">
      <div className="px-5 pt-4 pb-3 border-b border-border">
        <div className="flex items-center gap-2 mb-2">
          <span
            className={`font-mono text-[10px] uppercase tracking-wider px-2 py-0.5 rounded border ${
              data.mode === "cooperate" ? "border-sky-500 text-sky-600 dark:text-sky-400" : "border-border-strong text-muted-fg"
            }`}
          >
            {data.mode} mode
          </span>
          <span className="font-mono text-[11px] text-muted-fg">{data.agents.length} agents</span>
        </div>
        <div className="grid grid-cols-4 gap-2 font-mono text-[11px]">
          {(Object.keys(KINDS) as InteractionKind[]).map((k) => (
            <div key={k} className="rounded-md bg-muted px-2 py-1.5">
              <div className="text-base text-foreground" style={{ color: KINDS[k][theme] }}>{k === "broadcast" ? visible.messages.filter((m) => m.to === "all").length : k === "message" ? visible.messages.filter((m) => m.to !== "all").length : counts(k)}</div>
              <div className="text-muted-fg">{KINDS[k].label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between px-5 py-2 border-b border-border">
        <div className="flex font-mono text-[10px] uppercase tracking-wider border border-border rounded-md overflow-hidden">
          {(["messages", "all"] as const).map((f) => (
            <button
              key={f}
              onClick={() => onFeed(f)}
              className={`px-2.5 py-1 transition-colors ${feed === f ? "bg-foreground text-background" : "text-muted-fg hover:bg-muted"}`}
            >
              {f === "all" ? "all links" : f}
            </button>
          ))}
        </div>
        {selection && (
          <button onClick={onClear} className="font-mono text-[10px] text-muted-fg hover:text-foreground">
            {"agent" in selection ? selection.agent : `${selection.from} → ${selection.to}`} ✕
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-3 space-y-3">
        {items.length === 0 && (
          <p className="py-8 text-center font-mono text-xs text-muted-fg">
            {data.mode === "compete" && feed === "messages"
              ? "Compete mode: agents can't message each other. Switch to “all links” to see influence through shared memory."
              : "Nothing yet."}
          </p>
        )}
        {items.map((i, n) => (
          <div key={`${i.t}-${n}`} className="border-l-2 pl-3" style={{ borderColor: KINDS[i.kind][theme] }}>
            <div className="flex items-baseline gap-2 font-mono text-[10px] text-muted-fg">
              <span>{elapsed(i.t, t0)}</span>
              <span className="text-foreground">{i.from}</span>
              <span>→</span>
              <span className="text-foreground">{i.kind === "broadcast" ? "all" : i.to}</span>
              <span style={{ color: KINDS[i.kind][theme] }}>{KINDS[i.kind].label}</span>
            </div>
            <p className="text-[13px] mt-0.5 whitespace-pre-wrap break-words">{i.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
