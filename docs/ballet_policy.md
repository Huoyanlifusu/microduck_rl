# Ballet V1 policy

`Mjlab-Ballet-Flat-MicroDuck` is the first training scaffold for a ballet-like
one-legged pirouette. It is intentionally narrow: the right leg remains the
support/pivot leg, the left leg is the display leg, and each commanded turn is
a counter-clockwise 0.4 rad/s for 2.0-2.6 seconds (about 46-60 degrees).

## Command and deployment contract

The policy keeps the shared 61-observation / 14-action contract. Its three
twist values mean:

```text
[active, free_leg_side, turn]
```

- `[1, 1, 0.4]` lifts the left leg and tracks a 0.4 rad/s yaw rate.
- `[0, 1, 0]` asks the same policy to land and return to the two-foot HOME pose.

TURN and IDLE windows alternate every 2.0-2.6 seconds in training. Thus every
finite turn is followed by a learned brake, leg lowering, and HOME recovery.
Random window lengths prevent the policy from memorising one exact stop angle.

After exporting `ballet.onnx`, a local file can be installed as a generic
timed skill without adding a new daemon RPC:

```bash
sudo robotctl policy add ballet /path/to/ballet.onnx \
  --hold 2.3 --command 1,1,0.4 \
  --unwind 2.5 --unwind-command 0,1,0
robotctl robot do ballet
```

The timing above is illustrative. Measure it in simulation and on a supported
robot before treating it as a deployment value.

## Training progression

The initial reward stack first pays for right-foot-only support, left-foot
clearance and the display pose. Losing right-foot contact or touching down with
the left foot is explicitly penalised. At iteration 300 yaw-rate tracking and
its constant-gradient L1 bootstrap turn on; the main tracking reward reaches
full weight at iteration 600:

```text
right-foot support + left foot raised + trunk yaw rate near 0.4 rad/s
```

The same rate objective targets exactly zero when the command switches off, so
braking is trained rather than left to a fixed timeout. Turning credit is also
multiplied by an upright-posture score, preventing lean-and-thrash shortcuts.
A weak planar-velocity
cost discourages travelling across the floor, while leaving enough freedom for
the trunk to orbit slightly around the offset support foot. The all-axis angular
momentum penalty is removed because it would directly oppose yaw rotation.

Existing BallKick curricula continue to phase in CoM/head-CoM randomization and
pushes. From iteration 600 onward, upright and body-angular-velocity constraints
tighten and a small leg-action-acceleration penalty is introduced. Ballet caps
the final action-rate penalty at -0.5 so necessary balance and yaw corrections
remain viable without rewarding high-frequency jitter.

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
- No true pointe motion: Microduck has no toe joint; the support sole must pivot
  or slip against the ground model.
- No left/right unified policy, reverse turn or multi-step choreography yet.
- Reward/config tests and a short training smoke test verify wiring, not learned
  skill quality or hardware safety.

First hardware trials belong on a support rig, with a reduced action scale,
lower turn rate and short hold. Inspect support-foot slip, tilt, joint tracking
error, battery sag and servo temperature before increasing speed or duration.
