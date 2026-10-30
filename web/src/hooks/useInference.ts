import { useEffect, useState } from "react";
import { api, type InferenceResponse } from "../lib/api";

/** Poll /api/inference. Stops polling when the dashboard has no sampler. */
export function useInference(pollMs = 2000): InferenceResponse | null {
  const [data, setData] = useState<InferenceResponse | null>(null);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = () => {
      api
        .inference()
        .then((d) => {
          if (stopped) return;
          setData(d);
          if (d.enabled) timer = setTimeout(tick, pollMs);
        })
        .catch(() => {
          if (!stopped) timer = setTimeout(tick, pollMs * 5);
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
