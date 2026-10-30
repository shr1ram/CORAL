import { useEffect, useRef } from "react";

type SSEHandler = (data: unknown) => void;

const EVENT_TYPES = [
  "attempt:new",
  "attempt:update",
  "note:update",
  "log:update",
  "eval:update",
  "message:update",
];

/* One EventSource for the whole page, shared by every useSSE caller: opened
   with the first subscriber, closed with the last. */
const subscribers = new Set<{ current: Record<string, SSEHandler> }>();
let source: EventSource | null = null;

function open(): EventSource {
  const es = new EventSource("/api/events");
  for (const type of EVENT_TYPES) {
    es.addEventListener(type, (e) => {
      let data: unknown;
      try {
        data = JSON.parse((e as MessageEvent).data);
      } catch {
        return; // ignore parse errors
      }
      for (const sub of subscribers) sub.current[type]?.(data);
    });
  }
  // Cross-tab sync: reload when another tab switches runs
  es.addEventListener("run:switched", () => window.location.reload());
  es.onerror = () => {};
  return es;
}

export function useSSE(handlers: Record<string, SSEHandler>) {
  const handlersRef = useRef(handlers);
  useEffect(() => {
    handlersRef.current = handlers;
  });

  useEffect(() => {
    subscribers.add(handlersRef);
    source ??= open();
    return () => {
      subscribers.delete(handlersRef);
      if (subscribers.size === 0) {
        source?.close();
        source = null;
      }
    };
  }, []);
}
