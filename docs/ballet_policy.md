# Ballet V1 policy (A3)

`Mjlab-Ballet-Flat-MicroDuck` is the first training scaffold for a ballet-like
one-legged pirouette. It is intentionally narrow: the right leg remains the
support/pivot leg, the left leg is the display leg, and A3 prioritizes learning
a stable one-leg pose before introducing yaw.

## Command and deployment contract

The policy keeps the shared 61-observation / 14-action contract. Its three
twist values mean:

```text
[active, free_leg_side, turn]
```

A3 trains the fixed command `[1, 1, 0.4]` for each full 12-second episode:
active skill, left leg free, right leg supporting, and a target yaw rate of
0.4 rad/s. There is no TURN/IDLE resampling. At deployment the runtime holds
this policy for a measured duration and then switches back to the normal
standing policy. A 2.0-2.6 second hold would request about 46-60 degrees once
the turn has converged.

The four neck/head actions remain in the shared 14-action output, but Ballet
A3 holds them at HOME throughout the episode. The turn reward is multiplied by a
strict four-joint neck-pose score, so lowering the head as a counterweight can
no longer be traded for yaw reward. Dense trunk-height and upright terms likewise
prevent a deep crouch. This is still a planted-right-foot pirouette, not a
jump: loss of support-foot contact remains a violation.

After exporting `ballet.onnx`, install it as a timed skill. The exact command
depends on the corresponding `robotd` policy-switching interface; the important
sequence is:

```text
standing policy -> ballet.onnx + [1, 1, 0.4] for a short hold -> standing policy
```

The timing above is illustrative. Measure it in simulation and on a supported
robot before treating it as a deployment value.

## Why A3 adds weight transfer

Forward kinematics of the original free-leg joint target showed that the left
foot can reach its requested height, but the whole-robot CoM remains about
49 mm away from the right support foot. The old binary contact reward only
reported success after the foot had lifted; it gave no direction for the
preceding lateral weight shift. A2 therefore learned to stand straighter by
sacrificing the leg lift.

A3 adds a Gaussian reward for accurate final CoM placement and an L1 companion
that keeps a useful gradient while the CoM is still far away. The same pattern
is used for trunk height and the free-leg pose. Neck reward is reduced so it
cannot dominate the actual one-leg task.

## Training progression

Iterations 0-799 train right-foot support, lateral CoM transfer, left-foot
clearance, the free-leg pose, standing height and upright posture. Losing
right-foot contact or touching down with the left foot is explicitly penalised.
Yaw reward is exactly zero in this phase. At iteration 800 yaw-rate tracking
and its constant-gradient L1 bootstrap turn on; the main tracking reward
reaches full weight at iteration 1200:

```text
right-foot support + left foot raised + trunk yaw rate near 0.4 rad/s
```

Turning credit is multiplied by an upright-posture and neck-pose score,
preventing lean-and-thrash shortcuts. A weak planar-velocity
cost discourages travelling across the floor, while leaving enough freedom for
the trunk to orbit slightly around the offset support foot. The all-axis angular
momentum penalty is removed because it would directly oppose yaw rotation.

Stronger CoM/head-CoM randomization and pushes are delayed until after the
one-leg pose and first turn stage. From iteration 800 onward, upright and
body-angular-velocity constraints tighten; action smoothing is introduced
progressively from iteration 800 onward. Ballet caps
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

Watch `com_over_support`, `com_over_support_l1`, `free_foot_height`,
`free_leg_pose`, `unique_support`, `height_stand_l1`, `upright_linear`, contact
violation and fall rate. Before iteration 800, yaw reward should remain zero.

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
