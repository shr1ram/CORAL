## Plateau: change direction

You have not improved in several evals. This is a signal to change direction, not to stop.

1. Diagnose: `coral log --agent {agent_id}` and `coral log -n 10`. If everyone is stuck at the
   same score, the team shares a local basin.
2. Pick the highest-payoff idea nobody has really built: check `{shared_dir}/notes/index.md`,
   `_synthesis/` and `_open-questions.md`. Ideas dismissed without being built are still open.
   Filling a posture the team lacks (see roles and focus notes) is also a good pivot.
3. Claim it with a focus note (`create-notes` Variant C): posture, lane, budget, a concrete
   abandon_if, and why it should pay off, citing a note or attempt.
4. `coral checkout <best hash>`, then give it at least 3 real evals ("structural attempt 1/3 on
   X"). Use `--tune` only for sweeps inside the new direction.
