# Ballet V1 policy (A4)

`Mjlab-Ballet-Flat-MicroDuck` is the first training scaffold for a ballet-like
one-legged pirouette. It is intentionally narrow: the right leg remains the
support/pivot leg, the left leg is the display leg, and A4 assigns every one of
the 14 joints an explicit role before introducing body yaw.

## Command and deployment contract

The policy keeps the shared 61-observation / 14-action contract. Its three
twist values mean:

```text
[active, free_leg_side, turn]
```

A4 holds `[1, 1, 0]` through iteration 1200, then changes both the observed
command and its reward to 0.2 rad/s, reaching `[1, 1, 0.4]` at iteration 1600.
There is no TURN/IDLE resampling. At deployment the runtime holds this policy
for a measured duration and then switches back to the normal standing policy.

The four neck/head actions remain in the shared 14-action output, but position,
raw action and velocity penalties hold them at HOME. The turn reward is also
multiplied by strict neck, upright and one-foot-contact gates. Dense trunk-height
and upright terms prevent a deep crouch. This is a planted-right-foot
pirouette, not a
jump: loss of support-foot contact remains a violation.

After exporting `ballet.onnx`, install it as a timed skill. The exact command
depends on the corresponding `robotd` policy-switching interface; the important
sequence is:

```text
standing policy -> ballet.onnx + [1, 1, 0.4] for a short hold -> standing policy
```

The timing above is illustrative. Measure it in simulation and on a supported
robot before treating it as a deployment value.

## Joint roles

| Action indices | Joints | Role |
|---|---|---|
| 0 | left hip-yaw | Hold HOME; opening it is unnecessary |
| 1-4 | left roll/pitch/knee/ankle | Form and hold the lifted-leg pose |
| 5-8 | neck/head | Hold HOME; never generate body yaw with the head |
| 9 | right hip-yaw | Hold HOME before iteration 1200; body-turn actuator afterwards |
| 10-13 | right roll/pitch/knee/ankle | Free only for support and balance |

## Why A4 targets trunk transfer

Forward kinematics of the original free-leg joint target showed that the left
foot can reach its requested height, but the whole-robot CoM remains about
49 mm away from the right support foot. The old binary contact reward only
reported success after the foot had lifted; it gave no direction for the
preceding lateral weight shift. A2 therefore learned to stand straighter by
sacrificing the leg lift.

A3's whole-robot CoM reward could be improved by moving Microduck's heavy head.
A4 replaces it with Gaussian and L1 rewards that place the trunk-base projection
over the right foot. This requires hip/pelvis translation rather than head
swinging. The same dense pattern remains for trunk height and free-leg pose.

## Training progression

Iterations 0-1199 train right-foot support, lateral trunk transfer, left-foot
clearance, the free-leg pose, standing height and upright posture. Losing
right-foot contact or touching down with the left foot is explicitly penalised.
Both observed turn command and yaw reward are exactly zero in this phase. At
iteration 1200 they synchronously turn on at 0.2 rad/s; at iteration 1600 the
command reaches 0.4 rad/s and the main tracking reward reaches full weight:

```text
right-foot support + left foot raised + trunk yaw rate near 0.4 rad/s
```

Turning credit is multiplied by an upright-posture and neck-pose score,
preventing lean-and-thrash shortcuts. A weak planar-velocity
cost discourages travelling across the floor, while leaving enough freedom for
the trunk to orbit slightly around the offset support foot. The all-axis angular
momentum penalty is removed because it would directly oppose yaw rotation.

Stronger CoM/head-CoM randomization and pushes are delayed until after the
one-leg pose and first turn stage. Posture and action smoothing also tighten
only after the initial single-support behavior forms. Ballet caps
the final action-rate penalty at -0.5 so necessary balance and yaw corrections
remain viable without rewarding high-frequency jitter.

```bash
# Cheap configuration/runtime smoke test first.
uv run train Mjlab-Ballet-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5

# Then a fresh real run. Do not resume A1/A2: the command distribution and
# reward objective changed.
uv run train Mjlab-Ballet-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max_iterations 6000

uv run play Mjlab-Ballet-Flat-MicroDuck \
  --wandb-run-path <entity/project/run_id>

uv run scripts/export.py Mjlab-Ballet-Flat-MicroDuck \
  --wandb-run-path <entity/project/run_id> --onnx-file ballet.onnx
```

Watch `trunk_over_support`, `trunk_over_support_l1`, `free_foot_height`,
`free_leg_pose`, `unique_support`, `height_stand_l1`, `upright_linear`, contact
violation, neck penalties and fall rate. Before iteration 1200, both
`Curriculum/turn_command` and yaw reward must remain zero.

## What V1 does not claim

- No policy weights are included; this change supplies a trainable environment.
- No true pointe motion: Microduck has no toe joint; the support sole must pivot
  or slip against the ground model.
- No left/right unified policy, reverse turn or multi-step choreography yet.
- Returning to two-foot stand is delegated to the proven standing policy rather
  than learned by this first ballet policy.
- Reward/config tests and a short training smoke test verify wiring, not learned
  skill quality or hardware safety.

First hardware trials belong on a support rig, with a reduced action scale,
lower turn rate and short hold. Inspect support-foot slip, tilt, joint tracking
error, battery sag and servo temperature before increasing speed or duration.
