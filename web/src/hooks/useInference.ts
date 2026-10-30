import { useEffect, useState } from "react";
import { api, type InferenceResponse, type InferenceSample } from "../lib/api";
import { pollWhileVisible } from "../lib/visibility";

/** Poll /api/inference for the latest sample only. Stops polling when the
 *  dashboard has no sampler; paused while the tab is hidden. */
export function useInference(pollMs = 2000): InferenceResponse | null {
  const [data, setData] = useState<InferenceResponse | null>(null);

  useEffect(() => {
    let stopped = false;
    const stop = pollWhileVisible(async () => {
      const d = await api.inference();
      if (!stopped) setData(d);
      return d.enabled ? pollMs : null;
    }, pollMs);
    return () => {
      stopped = true;
      stop();
    };
  }, [pollMs]);

  return data;
}

/** The run's throughput history: fetched once (downsampled), then only the
 *  samples newer than the last one seen are appended. */
export function useInferenceHistory(pollMs = 5000): InferenceSample[] {
  const [samples, setSamples] = useState<InferenceSample[]>([]);

  useEffect(() => {
    let stopped = false;
    let last: number | null = null;
    const stop = pollWhileVisible(async () => {
      const d = await api.inference(last == null ? { history: true } : { since: last });
      if (stopped) return null;
      const fresh = d.samples ?? [];
      if (last == null) setSamples(fresh);
      else if (fresh.length) setSamples((s) => [...s, ...fresh]);
      if (fresh.length) last = fresh[fresh.length - 1].t;
      else if (last == null) last = 0;
      return d.enabled ? pollMs : null;
    }, pollMs);
    return () => {
      stopped = true;
      stop();
    };
  }, [pollMs]);

  return samples;
}
