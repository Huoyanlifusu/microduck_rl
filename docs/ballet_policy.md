# Ballet V1 policy

`Mjlab-Ballet-Flat-MicroDuck` is the first training scaffold for a ballet-like
one-legged low hop. It is intentionally narrow: the right leg supports and
hops, the left leg is the display leg, and yaw turning is disabled.

## Command and deployment contract

The policy keeps the shared 61-observation / 14-action contract. Its three
twist values mean:

```text
[active, free_leg_side, turn]
```

- `[1, 1, 0]` enters and maintains the V1 left-leg-up hop.
- `[0, 1, 0]` asks the same policy to land and return to the two-foot HOME pose.

The active flag is resampled every 3–6 seconds in training. This is deliberate:
robotd may begin the unwind at any hop phase, including flight. A policy that
only saw episode-boundary stops would not have a trained exit from that state.

After exporting `ballet.onnx`, a local file can be installed as a generic
perpetual skill without adding a new daemon RPC:

```bash
sudo robotctl policy add ballet /path/to/ballet.onnx \
  --hold 5 --command 1,1,0 \
  --unwind 2.5 --unwind-command 0,1,0
robotctl robot do ballet
```

The timing above is illustrative. Measure it in simulation and on a supported
robot before treating it as a deployment value.

## Training progression

The initial reward stack first pays for right-foot-only support, left-foot
clearance and the display pose. At iteration 300 the low trunk-height target
starts ramping in. At iteration 600 the stateful hop completion reward starts;
it pays only for:

```text
right support -> both feet airborne -> right support, left foot still clear
```

Free-foot contact invalidates an attempt. The latch is cleared on command-off,
so landing during unwind cannot farm hop reward. Existing BallKick curricula
continue to phase in action smoothness, CoM/head-CoM randomization and pushes.

```bash
# Cheap configuration/runtime smoke test first.
uv run train Mjlab-Ballet-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5

# Then a real run.
uv run train Mjlab-Ballet-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max_iterations 6000

uv run play Mjlab-Ballet-Flat-MicroDuck \
  --wandb-run-path <entity/project/run_id>

uv run scripts/export.py Mjlab-Ballet-Flat-MicroDuck \
  --wandb-run-path <entity/project/run_id> --onnx-file ballet.onnx
```

## What V1 does not claim

- No policy weights are included; this change supplies a trainable environment.
- No true pointe motion: Microduck has no toe joint.
- No left/right unified policy or yaw choreography yet.
- Reward/config tests and a short training smoke test verify wiring, not learned
  skill quality or hardware safety.

First hardware trials belong on a support rig, with a reduced action scale and
short hold. Inspect landing impact, joint tracking error, battery sag and servo
temperature before increasing hop height or duration.
