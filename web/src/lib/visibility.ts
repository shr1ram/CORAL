/** Poll `tick` every `pollMs` while the tab is visible; pause while hidden and
 *  poll once immediately when it becomes visible again. `tick` resolves to the
 *  delay before the next poll (null stops polling). */
export function pollWhileVisible(tick: () => Promise<number | null>, pollMs: number): () => void {
  let stopped = false;
  let inFlight = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const run = () => {
    clearTimeout(timer);
    if (stopped || inFlight || document.hidden) return;
    inFlight = true;
    tick()
      .catch(() => pollMs * 5)
      .then((next) => {
        inFlight = false;
        if (!stopped && next != null) timer = setTimeout(run, next);
      });
  };
  const onVisibility = () => {
    if (!document.hidden) run();
  };

  document.addEventListener("visibilitychange", onVisibility);
  run();
  return () => {
    stopped = true;
    clearTimeout(timer);
    document.removeEventListener("visibilitychange", onVisibility);
  };
}
