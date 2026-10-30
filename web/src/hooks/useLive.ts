import { useEffect, useState } from "react";
import { api, type LiveResponse } from "../lib/api";
import { pollWhileVisible } from "../lib/visibility";

/** Poll /api/live: what each agent is doing now. Paused while the tab is hidden. */
export function useLive(pollMs = 2000): LiveResponse | null {
  const [data, setData] = useState<LiveResponse | null>(null);

  useEffect(() => {
    let stopped = false;
    const stop = pollWhileVisible(async () => {
      const d = await api.live();
      if (!stopped) setData(d);
      return pollMs;
    }, pollMs);
    return () => {
      stopped = true;
      stop();
    };
  }, [pollMs]);

  return data;
}
