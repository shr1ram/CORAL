import { useEffect, useState } from "react";
import { api, type LiveResponse } from "../lib/api";

/** Poll /api/live: what each agent is doing now. */
export function useLive(pollMs = 2000): LiveResponse | null {
  const [data, setData] = useState<LiveResponse | null>(null);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = () => {
      api
        .live()
        .then((d) => !stopped && setData(d))
        .catch(() => {})
        .finally(() => {
          if (!stopped) timer = setTimeout(tick, pollMs);
        });
    };
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [pollMs]);

  return data;
}
