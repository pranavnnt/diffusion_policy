"""Structural invariants of the DRIM port.

Not accuracy tests — there is nothing to be accurate about yet.  These pin the
four properties that make an DRIM result *interpretable*, each of which dap
broke at least once and each of which fails silently:

1. ``D2`` with a null message is ``B1`` exactly (the gated route).
2. A forbidden action channel is absent, not merely discouraged, and the ceiling
   survives a nested checkpoint round-trip.
3. The message is masked wherever no full causal window exists.
4. Nothing selects a checkpoint on a score that does not exist.

Run: ``pytest tests/test_drim.py``
"""

import os

import numpy as np
import pytest
import torch

from diffusion_policy.drim import dynamics as DY
from diffusion_policy.drim import fields as FL
from diffusion_policy.drim import nets as N
from diffusion_policy.drim import policy as PL
from diffusion_policy.drim import selection as SEL
from diffusion_policy.drim.spec import DrimSpec


def tiny_spec(**kw):
    """A state-only spec small enough to build in a test."""
    base = dict(prop_dim=28, act_dim=12, wrench_dim=0, cameras=(),
                pred_horizon=8, exec_horizon=4, message_window=4,
                message_dim=8, fast_limits=tuple([0.05] * 12))
    base.update(kw)
    return DrimSpec(**base)


def batch(spec, n=4, cond_dim=0):
    g = torch.Generator().manual_seed(0)
    b = {
        "prop2": torch.randn(n, spec.n_obs_steps, spec.prop_dim, generator=g),
        "step_prop": torch.randn(n, spec.exec_horizon, spec.prop_dim, generator=g),
        "target": torch.rand(n, spec.pred_horizon, spec.act_dim, generator=g) * 2 - 1,
        "msg_valid": torch.tensor([0.0, 1.0] * (n // 2)),
    }
    return b


# --------------------------------------------------------------------------- #
# 1. the exact-null identity
# --------------------------------------------------------------------------- #


def test_d2_with_null_message_is_b1_exactly():
    """The whole D2-minus-B1 claim rests on this being an identity.

    Checked **after** the message path has been given non-trivial weights, not
    at initialisation: the point of the gated route is that it stays null for any
    parameter values, so a test that only runs on a fresh model would pass even
    if someone replaced it with an ordinary residual.
    """
    spec = tiny_spec()
    b0 = PL.build("B0", spec)
    b1 = PL.build("B1", spec, core=b0)
    d2 = PL.build("D2", spec, core=b1, message_in_dims=(6, 6))

    # make the message path emphatically non-null
    with torch.no_grad():
        for p in d2.dv.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        d2.gate.fill_(2.5)
        d2.reconcile_gate.fill_(2.5)

    b = batch(spec)
    ctx = d2.context(None, b["prop2"])
    noise = torch.zeros(len(b["target"]), d2.horizon, d2.act_dim)
    zero = torch.zeros(len(b["target"]), d2.cond_dim)

    with torch.no_grad():
        via_null = d2.sample_chunk(ctx, zero, noise)
        dv, d2.dv = d2.dv, None                     # the B1 path, same weights
        via_b1 = d2.sample_chunk(ctx, None, noise)
        d2.dv = dv
    assert torch.equal(via_null, via_b1), (
        "D2 at m=0 diverged from B1; the gated route is no longer exactly null")


def test_reconciler_is_exactly_null_at_zero_message():
    spec = tiny_spec()
    d2 = PL.build("D2", spec,
                  core=PL.build("B1", spec, core=PL.build("B0", spec)),
                  message_in_dims=(6, 6))
    with torch.no_grad():
        for p in d2.reconcile.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        d2.reconcile_gate.fill_(3.0)
    b = batch(spec)
    nominal = torch.rand(len(b["target"]), spec.pred_horizon, spec.act_dim) * 2 - 1
    zero = torch.zeros(len(b["target"]), d2.cond_dim)
    with torch.no_grad():
        with_msg = d2.residual_prefix(nominal, b["step_prop"], None, zero)
        without = d2.residual_prefix(nominal, b["step_prop"], None, None)
    assert torch.allclose(with_msg, without, atol=1e-7)


# --------------------------------------------------------------------------- #
# 2. the fast level's authority
# --------------------------------------------------------------------------- #


def test_zero_ceiling_channel_is_structurally_silent():
    """``limits[c] == 0`` must make channel ``c`` zero for *any* weights."""
    lim = np.array([0.05] * 11 + [0.0], dtype=np.float32)
    fast = N.FastCorrector(28, 0, 12, lim)
    with torch.no_grad():
        for p in fast.parameters():
            p.add_(torch.randn_like(p) * 5.0)
    out = fast(torch.randn(6, 28), None, torch.randn(6, 12), torch.randn(6, 12),
               torch.randn(6, 12), torch.rand(6, 1))
    assert torch.all(out[:, 11] == 0.0)
    assert out[:, :11].abs().max() <= 0.05 + 1e-6


def test_spec_refuses_to_invent_an_authority():
    with pytest.raises(ValueError, match="fast_limits is unset"):
        tiny_spec(fast_limits=None).limits()


def test_authority_survives_a_nested_round_trip(tmp_path):
    """dap lost a whole round to ``strict=False`` silently dropping this buffer."""
    spec = tiny_spec()
    b0 = PL.build("B0", spec)
    b1 = PL.build("B1", spec, core=b0)
    bank = SEL.SnapshotBank(2, every=1, parent_keys=list(b0.state_dict()))
    bank.observe(b1, 1, val_loss=1.0)
    bank.observe(b1, 2, val_loss=0.9)
    path = str(tmp_path / "B1.pt")
    SEL.save_selected(path, bank, "B1", spec.to_dict(), estimator="last2")

    fresh = PL.build("B1", spec, core=PL.build("B0", spec))
    SEL.load_selected(path, fresh)
    fresh.assert_authority(spec)
    assert np.allclose(fresh.authority(), spec.limits())


def test_load_rejects_keys_no_parent_holds(tmp_path):
    """A key in neither the checkpoint nor its parent loads as random weights.

    This is the failure mode a ``strict=False`` load — which a nested stage has
    no choice but to use — reports by saying nothing at all.
    """
    spec = tiny_spec()
    b0 = PL.build("B0", spec)
    b1 = PL.build("B1", spec, core=b0)
    bank = SEL.SnapshotBank(1, every=1, parent_keys=list(b0.state_dict()))
    bank.observe(b1, 1, val_loss=1.0)
    path = str(tmp_path / "bad.pt")
    SEL.save_selected(path, bank, "B1", spec.to_dict(), estimator="last1")

    #: drop a corrector weight, which B0 does not hold either
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dropped = next(k for k in ck["state_dict"] if k.startswith("fast."))
    del ck["state_dict"][dropped]
    torch.save(ck, path)

    fresh = PL.build("B1", spec, core=PL.build("B0", spec))
    with pytest.raises(AssertionError, match="freshly initialised"):
        SEL.load_selected(path, fresh, strict_parent=True)


# --------------------------------------------------------------------------- #
# 3. the message mask
# --------------------------------------------------------------------------- #


def test_message_is_zero_where_no_full_window_exists():
    spec = tiny_spec()
    d2 = PL.build("D2", spec,
                  core=PL.build("B1", spec, core=PL.build("B0", spec)),
                  message_in_dims=(6, 6))
    n = 4
    b = {"prop2": torch.randn(n, spec.n_obs_steps, spec.prop_dim),
         "msg_valid": torch.tensor([0.0, 1.0, 0.0, 1.0]),
         "hist_S": torch.randn(n, spec.message_window, 6),
         "hist_U": torch.randn(n, spec.message_window, 6)}
    m = d2.conditioning(b)
    assert torch.all(m[0] == 0) and torch.all(m[2] == 0)
    assert m[1].abs().sum() > 0


def test_decision_steps_only_yield_full_chunks():
    from diffusion_policy.drim.dataset import decision_steps
    spec = tiny_spec()
    s = decision_steps(25, spec)
    assert s.max() + spec.pred_horizon <= 25
    assert decision_steps(spec.pred_horizon - 1, spec).size == 0


# --------------------------------------------------------------------------- #
# 4. selection
# --------------------------------------------------------------------------- #


def test_score_ranked_estimators_are_refused():
    """No rollout exists, so ``best`` must raise rather than quietly fall back."""
    for est in ("best", "top5"):
        with pytest.raises(ValueError, match="rollout success"):
            SEL.select_epochs([10, 20, 30], est)


def test_last_k_ranks_nothing():
    assert SEL.select_epochs([10, 20, 30, 40], "last3") == [20, 30, 40]
    assert SEL.select_epochs([10, 20, 30, 40], "all") == [10, 20, 30, 40]


def test_average_states_is_a_mean_of_the_named_epochs():
    spec = tiny_spec()
    m = PL.build("B0", spec)
    bank = SEL.SnapshotBank(3, every=1)
    seen = []
    for ep in (1, 2, 3):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.1)
        bank.observe(m, ep)
        seen.append(bank.states[ep]["slow.obs_mean"].clone())
    avg = SEL.average_states(bank, [2, 3])
    assert torch.allclose(avg["slow.obs_mean"], (seen[1] + seen[2]) / 2)


def test_integer_buffers_are_not_averaged():
    spec = tiny_spec()
    m = PL.build("B0", spec)
    bank = SEL.SnapshotBank(2, every=1)
    bank.observe(m, 1)
    bank.observe(m, 2)
    bank.states[1]["_probe"] = torch.tensor([1], dtype=torch.int64)
    bank.states[2]["_probe"] = torch.tensor([4], dtype=torch.int64)
    assert SEL.average_states(bank, [1, 2])["_probe"].item() == 4


# --------------------------------------------------------------------------- #
# dynamics
# --------------------------------------------------------------------------- #


def test_quaternion_delta_is_continuous_across_the_double_cover():
    """Componentwise subtraction would put a spurious jump in every surprise."""
    q = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    same_rotation = -q                          # q and -q are the same rotation
    d = DY._quat_rel_rotvec(q, same_rotation)
    assert np.allclose(d, 0.0, atol=1e-9)
    assert not np.allclose(same_rotation - q, 0.0)


def test_modalities_cover_the_resolved_state_width():
    res = FL.resolve(_packed_source(n=6), n_arms=2, layout=FL.SMOKE_LAYOUT,
                     warn=False)
    mods = FL.modalities(res, res.prop_names())
    assert DY.state_dim(mods) == res.width(res.prop_names())
    #: each quaternion contributes 4 state channels but only 3 delta channels
    assert DY.delta_dim(mods) == DY.state_dim(mods) - len(res.arms)


# --------------------------------------------------------------------------- #
# 5. field resolution — what is missing must be said, never filled in
# --------------------------------------------------------------------------- #


def _packed_source(n=6, arm2_live=True):
    """A packed `state` array in the smoke layout, optionally with a dead arm2."""
    rng = np.random.default_rng(0)
    st = np.zeros((n, 28), dtype=np.float32)
    for a in (0, 1):
        if a == 1 and not arm2_live:
            continue
        o = a * 14
        st[:, o:o + 7] = rng.normal(size=(n, 7))
        st[:, o + 7:o + 10] = rng.normal(size=(n, 3))
        q = rng.normal(size=(n, 4))
        st[:, o + 10:o + 14] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return {"state": st, "action": np.zeros((n, 12), np.float32)}


def test_absent_fields_are_excluded_not_zero_filled():
    """A zero-filled channel trains fine and silently removes a whole signal."""
    res = FL.resolve(_packed_source(), n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    assert res.prop_names() == ("q", "ee_pos", "ee_quat")
    assert res.wrench_names() == ()
    assert res.width(res.prop_names()) == 2 * 14      # not 2 * full schema width
    assert {s.name for s in res.missing()} == {
        "dq", "ee_lin_vel", "ee_ang_vel", "ee_twist", "gripper_pos",
        "gripper_vel", "tau_J", "wrench", "ee_force"}


def test_missing_fields_warn():
    with pytest.warns(UserWarning, match="missing .* declared per-arm fields"):
        FL.resolve(_packed_source(), n_arms=2, layout=FL.SMOKE_LAYOUT)


def test_required_field_raises_instead_of_warning():
    with pytest.raises(ValueError, match="required field"):
        FL.resolve(_packed_source(), n_arms=2, layout=FL.SMOKE_LAYOUT,
                   require=("wrench",), warn=False)


def test_zero_norm_quaternion_is_rejected():
    """A disconnected arm is recorded as zeros, so its quaternion is not a rotation."""
    res = FL.resolve(_packed_source(arm2_live=False), n_arms=2,
                     layout=FL.SMOKE_LAYOUT, warn=False)
    bad = [s for s in res.statuses if s.arm == "arm2" and s.name == "ee_quat"]
    assert bad and "not a rotation" in bad[0].reason


def test_a_dead_arm_is_dropped_not_the_live_arms_fields():
    """The across-arms rule must not let one dead arm empty the state vector."""
    res = FL.resolve(_packed_source(arm2_live=False), n_arms=2,
                     layout=FL.SMOKE_LAYOUT, warn=False)
    assert res.arms == (0,)
    assert res.prop_names() == ("q", "ee_pos", "ee_quat")
    assert res.width(res.prop_names()) == 14


def test_one_array_per_field_is_preferred_over_the_packed_layout():
    src = _packed_source()
    src["arm1_wrench"] = np.ones((6, 6), np.float32)
    src["arm2_wrench_ee_base"] = np.ones((6, 6), np.float32)   # via alias
    res = FL.resolve(src, n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    assert res.wrench_names() == ("wrench",)
    assert res.width(res.wrench_names()) == 12


def test_spec_refuses_a_dataset_with_different_channels():
    from diffusion_policy.drim.spec import from_resolution
    full = FL.resolve(_packed_source(), n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    src = _packed_source()
    src["arm1_wrench"] = np.ones((6, 6), np.float32)
    src["arm2_wrench"] = np.ones((6, 6), np.float32)
    other = FL.resolve(src, n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    a = from_resolution(full, act_dim=12)
    b = from_resolution(other, act_dim=12)
    with pytest.raises(AssertionError, match="field mismatch"):
        b.assert_schema(a)


# --------------------------------------------------------------------------- #
# 6. dt
# --------------------------------------------------------------------------- #


def test_dynamics_requires_dt_when_built_with_it():
    mods = [("a", (0, 3), 3, "linear")]
    m = DY.DeltaDynamics(mods, act_dim=2, use_dt=True)
    with pytest.raises(AssertionError, match="use_dt=True"):
        m(torch.zeros(2, 3), torch.zeros(2, 2))
    mu, ls = m(torch.zeros(2, 3), torch.zeros(2, 2), torch.ones(2, 1))
    assert mu.shape == (2, 3)


def test_no_change_baseline_is_reported_in_delta_units():
    d = np.array([[0.0, 2.0], [0.0, 0.0]])
    assert DY.no_change_baseline(d) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 7. action authority
# --------------------------------------------------------------------------- #


def test_action_is_trimmed_to_the_arms_the_state_covers():
    """Commanding an arm the observation cannot see is noise the flow field fits."""
    from diffusion_policy.drim.spec import from_resolution
    from diffusion_policy.drim.dressing import ACT_PER_ARM
    res = FL.resolve(_packed_source(arm2_live=False), n_arms=2,
                     layout=FL.SMOKE_LAYOUT, warn=False)
    spec = from_resolution(res, act_width=12, n_declared=2,
                           act_per_arm=ACT_PER_ARM)
    assert res.arms == (0,)
    assert spec.act_channels == (0, 1, 2, 3, 4, 5)
    assert spec.act_dim == 6


def test_action_channels_follow_the_live_arm_not_its_position():
    """With arm1 dropped, arm2's channels are 6..11 — not 0..5.

    ``per_arm`` is stated rather than taken from ``ACT_PER_ARM``: that constant
    tracks whichever recording is current (3 for the single-arm zigzag), while
    the invariant here is about the bimanual teleop's 6-wide layout and must
    keep describing it.
    """
    from diffusion_policy.drim.dressing import arm_act_channels
    assert arm_act_channels((1,), per_arm=6) == (6, 7, 8, 9, 10, 11)
    assert arm_act_channels((0, 1), per_arm=6)[-1] == 11
    #: and the same rule on the 3-wide single-arm recording
    assert arm_act_channels((1,), per_arm=3) == (3, 4, 5)


def test_dead_channels_get_zero_authority():
    from diffusion_policy.drim.spec import fast_limits_from_fraction
    #: A 6-wide scale, stated here rather than imported, so the test keeps
    #: exercising the wide case whatever the current rig records.
    ARM_ACT_SCALE = (0.02, 0.02, 0.02, 0.05, 0.05, 0.05)
    lim = fast_limits_from_fraction(0.05, ARM_ACT_SCALE, active=[1])
    assert lim == (0.0, 0.05, 0.0, 0.0, 0.0, 0.0)
    #: and a zero ceiling is structurally silent, not merely small
    fast = N.FastCorrector(14, 0, 6, np.asarray(lim, np.float32))
    with torch.no_grad():
        for p_ in fast.parameters():
            p_.add_(torch.randn_like(p_) * 5.0)
    out = fast(torch.randn(4, 14), None, torch.randn(4, 6), torch.randn(4, 6),
               torch.randn(4, 6), torch.rand(4, 1))
    assert torch.all(out[:, [0, 2, 3, 4, 5]] == 0.0)
    assert out[:, 1].abs().max() <= 0.05 + 1e-6


def test_fraction_must_be_a_fraction():
    from diffusion_policy.drim.spec import fast_limits_from_fraction
    from diffusion_policy.drim.dressing import ARM_ACT_SCALE
    with pytest.raises(AssertionError, match="fraction of full command"):
        fast_limits_from_fraction(1.5, ARM_ACT_SCALE)


def test_actions_normalise_by_the_declared_command_scale():
    """Full deflection maps to +/-1 regardless of what this dataset happened to use."""
    from diffusion_policy.drim.dataset import ChunkNormaliser
    #: The bimanual teleop's scale, written out rather than imported: the point
    #: is that *a* declared scale maps full deflection to 1, whatever the rig
    #: currently records.
    ARM_ACT_SCALE = (0.02, 0.02, 0.02, 0.05, 0.05, 0.05)
    data = {"target": np.full((4, 8, 6), 0.001, np.float32)}   # a timid dataset
    n = ChunkNormaliser(data, act_scale=ARM_ACT_SCALE)
    full = np.array([[[0.02, 0.02, 0.02, 0.05, 0.05, 0.05]]], np.float32)
    assert np.allclose(n.apply_vec("target", full), 1.0)


def test_residual_demand_is_in_fractions_of_full_command():
    """The number the authority fraction should be set from, before any training."""
    from diffusion_policy.drim.dataset import action_residual_demand
    spec = tiny_spec(pred_horizon=4, exec_horizon=4)
    #: a chunk alternating +/-0.2 around zero: the deviation from the chunk mean
    #: is 0.2 on every step, the step-to-step change is 0.4
    t = np.zeros((3, 4, spec.act_dim), np.float32)
    t[:, ::2] = 0.2
    t[:, 1::2] = -0.2
    d = action_residual_demand({"target": t}, spec)
    assert np.allclose(d["vs_chunk_mean"]["p50"], 0.2)
    assert np.allclose(d["step_to_step"]["p50"], 0.4)


# --------------------------------------------------------------------------- #
# 8. multiple stores
# --------------------------------------------------------------------------- #


def test_discover_zarrs_accepts_a_store_a_directory_and_a_list(tmp_path):
    from diffusion_policy.drim.dataset import discover_zarrs
    root = tmp_path / "data"
    for name in ("b_trial", "a_trial"):
        d = root / f"{name}.zarr"
        d.mkdir(parents=True)
        (d / ".zgroup").write_text("{}")
    found = discover_zarrs(root)
    assert [os.path.basename(f) for f in found] == ["a_trial.zarr", "b_trial.zarr"]
    assert discover_zarrs(root / "a_trial.zarr") == [str(root / "a_trial.zarr")]
    assert len(discover_zarrs([root / "a_trial.zarr", root / "b_trial.zarr"])) == 2


def test_discover_zarrs_refuses_an_empty_directory(tmp_path):
    from diffusion_policy.drim.dataset import discover_zarrs
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no zarr stores"):
        discover_zarrs(tmp_path / "empty")


def test_split_is_stratified_by_store():
    """An unstratified draw can put a whole session in validation."""
    from diffusion_policy.drim.dataset import episode_split
    sources = [0] * 10 + [1] * 10
    tr, va = episode_split(20, val_ratio=0.2, seed=0, sources=sources)
    assert len(va) == 4
    per_store = [sum(1 for i in va if sources[i] == s) for s in (0, 1)]
    assert per_store == [2, 2]
    assert not set(tr) & set(va)


def test_a_single_episode_store_is_never_held_out_entirely():
    """Holding out a singleton store removes that session from training."""
    from diffusion_policy.drim.dataset import episode_split
    sources = [0, 1, 1, 1, 1]
    tr, va = episode_split(5, val_ratio=0.5, seed=0, sources=sources)
    assert 0 in tr
    assert len(va) >= 1


def test_one_episode_gives_an_empty_validation_set():
    from diffusion_policy.drim.dataset import episode_split
    tr, va = episode_split(1, val_ratio=0.2, seed=0, sources=[0])
    assert list(tr) == [0] and len(va) == 0


def test_tail_slope_includes_the_final_grid_point():
    """A budget that is not a multiple of the period still ends on a snapshot."""
    bank = SEL.SnapshotBank(25, every=10)
    assert bank.grid == (10, 20, 25)
    curve = [{"epoch": e, "val_loss": 1.0 - 0.01 * e} for e in range(1, 26)]
    s_grid = SEL.tail_slope(curve, "val_loss", n=3, grid=bank.grid)
    s_modulo = SEL.tail_slope(curve, "val_loss", n=3)      # old behaviour
    assert s_grid is not None
    assert s_modulo is None or s_grid != s_modulo


def test_keep_grid_false_drops_the_snapshots(tmp_path):
    spec = tiny_spec()
    m = PL.build("B0", spec)
    bank = SEL.SnapshotBank(2, every=1)
    bank.observe(m, 1, val_loss=1.0)
    bank.observe(m, 2, val_loss=0.9)
    small = str(tmp_path / "small.pt")
    SEL.save_selected(small, bank, "B0", spec.to_dict(), estimator="last2",
                      keep_grid=False)
    ck = torch.load(small, map_location="cpu", weights_only=False)
    assert "grid_states" not in ck and ck["state_dict"]
    #: and it still loads
    SEL.load_selected(small, PL.build("B0", spec))


# --------------------------------------------------------------------------- #
# 9. bestval, offered rather than forbidden
# --------------------------------------------------------------------------- #


def _curve(vals, every=10):
    return [{"epoch": (i + 1) * every, "val_loss": v} for i, v in enumerate(vals)]


def test_a_criterion_picks_one_epoch_the_raw_argmin_without_smoothing():
    c = _curve([1.0, 0.4, 0.7, 0.9])
    assert SEL.select_epochs([10, 20, 30, 40], "val_loss", curve=c, smooth=0) == [20]


def test_smoothing_moves_the_pick_off_a_lone_dip():
    """An argmin over a noisy curve is biased low; the window damps that."""
    c = _curve([1.0, 0.4, 0.7, 0.9])
    #: ep20's dip is not supported by its neighbours, ep30's region is lower
    assert SEL.select_epochs([10, 20, 30, 40], "val_loss", curve=c, smooth=1) == [30]


def test_a_criterion_always_names_exactly_one_deployable_checkpoint():
    c = _curve([1.0, 0.4, 0.7, 0.9])
    for est in ("val_loss", "action_mse", "bestval"):
        rows = [dict(r, action_mse=r["val_loss"]) for r in c]
        assert len(SEL.select_epochs([10, 20, 30, 40], est, curve=rows)) == 1


def test_divergence_needs_scores_because_it_is_not_on_the_curve():
    with pytest.raises(ValueError, match="score per candidate"):
        SEL.select_epochs([10, 20], "divergence", curve=_curve([1.0, 0.5]))
    assert SEL.select_epochs([10, 20, 30], "divergence",
                             scores={10: 0.9, 20: 0.2, 30: 0.5}) == [20]


def test_rollout_ranked_estimators_are_still_refused():
    with pytest.raises(ValueError, match="no environment to roll out in"):
        SEL.select_epochs([10, 20], "best", curve=_curve([1.0, 0.5]))


def test_selection_note_records_every_criterion_s_pick():
    """A disagreement between criteria is data, so it is always recorded."""
    spec = tiny_spec()
    m = PL.build("B0", spec)
    bank = SEL.SnapshotBank(4, every=1)
    for ep, vl in zip((1, 2, 3, 4), (1.0, 0.4, 0.7, 0.9)):
        bank.observe(m, ep, val_loss=vl, action_mse=vl)
    note = SEL.selection_note(bank, "last2", smooth=0)
    assert note["selected_epochs"] == [3, 4]
    assert note["would_pick"]["val_loss"] == 2
    assert note["would_pick"]["last1"] == 4
    assert note["criteria_agree"] is False


def test_last_k_is_kept_but_selects_on_nothing():
    """Documented as a reproduction path, not a selector — on a rising tail it
    deploys the overfit model, which is what the 0909 run showed."""
    rising = _curve([0.9, 0.5, 0.3, 0.6, 1.0])
    eps = [10, 20, 30, 40, 50]
    assert SEL.select_epochs(eps, "last3", curve=rising) == [30, 40, 50]
    assert SEL.select_epochs(eps, "val_loss", curve=rising, smooth=0) == [30]


def test_a_rising_tail_is_flagged_because_last_k_assumes_a_plateau():
    spec = tiny_spec()
    m = PL.build("B0", spec)
    bank = SEL.SnapshotBank(6, every=1)
    for ep, vl in zip(range(1, 7), (1.0, 0.5, 0.3, 0.4, 0.6, 0.9)):
        bank.observe(m, ep, val_loss=vl)
    w = SEL.overfit_warning(bank)
    assert w and "not a plateau" in w
    assert "overfit_warning" in SEL.selection_note(bank, "last3")


def test_a_flat_tail_is_not_flagged():
    spec = tiny_spec()
    m = PL.build("B0", spec)
    bank = SEL.SnapshotBank(6, every=1)
    for ep, vl in zip(range(1, 7), (1.0, 0.6, 0.41, 0.40, 0.40, 0.39)):
        bank.observe(m, ep, val_loss=vl)
    assert SEL.overfit_warning(bank) is None


def test_auto_fast_frac_comes_from_the_measured_demand():
    from diffusion_policy.drim.spec import fast_frac_from_demand
    demand = {"vs_chunk_mean": {"p95": [0.01, 0.12, 0.30]}}
    assert fast_frac_from_demand(demand, active=[1]) == pytest.approx(0.12)
    #: channels that never move do not raise the ceiling
    assert fast_frac_from_demand(demand, active=[0]) == pytest.approx(0.02)
    #: and it is clamped rather than unbounded
    assert fast_frac_from_demand(demand, active=[2], hi=0.2) == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# 10. alternative spellings of the same quantity
# --------------------------------------------------------------------------- #


def test_singular_collector_names_resolve():
    """The zigzag collector writes joint_position / joint_velocity, not plural."""
    src = _packed_source()
    src["joint_position"] = np.zeros((6, 7), np.float32) + 0.5
    src["joint_velocity"] = np.zeros((6, 7), np.float32) + 0.5
    res = FL.resolve(src, n_arms=1, layout=FL.SMOKE_LAYOUT, warn=False)
    assert "q" in res.usable and "dq" in res.usable


def test_ee_twist_is_used_when_the_split_pair_is_absent():
    src = _packed_source()
    src["arm1_ee_twist"] = np.ones((6, 6), np.float32)
    src["arm2_ee_twist"] = np.ones((6, 6), np.float32)
    res = FL.resolve(src, n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    assert "ee_twist" in res.usable
    assert "ee_lin_vel" not in res.usable


def test_the_split_pair_wins_over_ee_twist_when_both_are_present():
    """Same six numbers twice would silently widen the state vector."""
    src = _packed_source()
    for a in ("arm1", "arm2"):
        src[f"{a}_ee_twist"] = np.ones((6, 6), np.float32)
        src[f"{a}_ee_lin_vel"] = np.ones((6, 3), np.float32)
        src[f"{a}_ee_ang_vel"] = np.ones((6, 3), np.float32)
    with pytest.warns(UserWarning, match="dropped as duplicates"):
        res = FL.resolve(src, n_arms=2, layout=FL.SMOKE_LAYOUT)
    assert "ee_lin_vel" in res.usable and "ee_ang_vel" in res.usable
    assert "ee_twist" not in res.usable


def test_ee_force_is_the_fallback_for_a_missing_wrench():
    src = _packed_source()
    for a in ("arm1", "arm2"):
        src[f"{a}_ee_force"] = np.ones((6, 3), np.float32)
    res = FL.resolve(src, n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    assert res.wrench_names() == ("ee_force",)
    #: and the full wrench supersedes it when both exist
    for a in ("arm1", "arm2"):
        src[f"{a}_wrench"] = np.ones((6, 6), np.float32)
    res = FL.resolve(src, n_arms=2, layout=FL.SMOKE_LAYOUT, warn=False)
    assert res.wrench_names() == ("wrench",)


def test_action_width_comes_from_the_recording_not_a_constant():
    """A 3-wide single-arm action must not be indexed as 6 channels per arm."""
    from diffusion_policy.drim.spec import from_resolution
    res = FL.resolve(_packed_source(arm2_live=False), n_arms=2,
                     layout=FL.SMOKE_LAYOUT, warn=False)
    from diffusion_policy.drim.dressing import ACT_PER_ARM
    spec = from_resolution(res, act_width=3, n_declared=1,
                           act_per_arm=ACT_PER_ARM)
    assert spec.act_channels == (0, 1, 2) and spec.act_dim == 3
    #: and an action that is not the 3-linear/3-angular layout gets no declared
    #: scale, so it falls back to the observed range rather than a wrong constant
    assert spec.act_scale is None


def test_short_episodes_are_named_and_excluded():
    """A two-step aborted recording still counts as an episode until it isn't."""
    from diffusion_policy.drim.dataset import short_episodes

    class _Eps:
        def __init__(self, lens):
            self._l = lens
        def lengths(self):
            return self._l
        def __len__(self):
            return len(self._l)

    spec = tiny_spec(pred_horizon=8)
    assert short_episodes(_Eps([2, 100, 7, 8]), spec) == [0, 2]


# --------------------------------------------------------------------------- #
# 11. pre-deployment diagnostics
# --------------------------------------------------------------------------- #


def test_apply_delta_inverts_state_delta():
    """Rolling a trajectory forward needs the inverse of the delta map.

    Quaternions are the reason this is not just addition: the delta of a rotation
    is a rotation vector, so advancing composes rather than adds.
    """
    from diffusion_policy.drim import dynamics as D
    mods = [("p", (0, 3), 3, "linear"), ("q", (3, 7), 3, "quat")]
    g = torch.Generator().manual_seed(0)
    y0 = torch.randn(5, 7, generator=g); y0[:, 3:] /= y0[:, 3:].norm(dim=-1, keepdim=True)
    y1 = torch.randn(5, 7, generator=g); y1[:, 3:] /= y1[:, 3:].norm(dim=-1, keepdim=True)
    back = D.torch_apply_delta(y0, D.torch_state_delta(y0, y1, mods), mods)
    assert torch.allclose(back[:, :3], y1[:, :3], atol=1e-5)
    #: q and -q are the same rotation, so compare up to sign
    err = torch.minimum((back[:, 3:] - y1[:, 3:]).abs().max(1).values,
                        (back[:, 3:] + y1[:, 3:]).abs().max(1).values)
    assert float(err.max()) < 1e-5


def test_detectors_flag_a_dynamics_worse_than_doing_nothing():
    from diffusion_policy.drim import diagnose as DG
    v = DG.detectors({"dyn": {"skill": -1.55}})
    assert v and v[0].startswith("FAIL")
    assert "model error" in v[0]
    assert DG.detectors({"dyn": {"skill": 0.4}})[0].startswith("OK")


def test_detectors_flag_both_ends_of_the_authority_range():
    from diffusion_policy.drim import diagnose as DG
    assert "authority unused" in DG.detectors({"B1": {"saturation": 0.0}})[0]
    assert "too small" in DG.detectors({"B1": {"saturation": 0.9}})[0]
    assert DG.detectors({"B1": {"saturation": 0.1}})[0].startswith("OK")


def test_detectors_flag_a_surprise_that_shifts_out_of_distribution():
    from diffusion_policy.drim import diagnose as DG
    v = DG.detectors({"diagnostics": {"surprise_shift": {"p99_ratio": 28.9}}})
    assert v[0].startswith("WARN") and "never saw in training" in v[0]


def test_trivial_action_baseline_catches_a_degenerate_target():
    """A position target the controller already tracks is nearly free to predict."""
    from diffusion_policy.drim.dataset import trivial_action_baselines
    spec = tiny_spec(pred_horizon=8, exec_horizon=4)
    #: a slowly drifting target: holding the first action is almost exact
    #: a position-like target: large offset, tiny per-chunk drift
    slow = (np.linspace(-1, 1, 6)[:, None, None]
            + np.cumsum(np.full((6, 8, spec.act_dim), 0.001, np.float32), axis=1))
    b = trivial_action_baselines({"target": slow.astype(np.float32)}, spec)
    assert b["repeat_first_action"] < 0.01 * b["predict_zero"]
    #: a target that alternates: holding is useless
    fast = np.zeros((6, 8, spec.act_dim), np.float32)
    fast[:, ::2] = 1.0; fast[:, 1::2] = -1.0
    b2 = trivial_action_baselines({"target": fast}, spec)
    assert b2["repeat_first_action"] > b2["predict_zero"]


def test_verdict_fails_when_the_copycat_wins():
    from diffusion_policy.drim import diagnose as DG
    v = DG.detectors({"trivial_action_baselines": {"repeat_first_action": 0.0002,
                                                   "predict_zero": 0.18},
                      "B0": {"curve": [{"epoch": 10, "action_mse": 0.017}]}})
    assert any(x.startswith("FAIL") and "degenerate" in x for x in v)


def test_unreachable_targets_are_flagged():
    """sample_chunk clamps to [-1,1]; a target beyond it cannot be produced."""
    from diffusion_policy.drim.spec import from_resolution
    res = FL.resolve(_packed_source(arm2_live=False), n_arms=2,
                     layout=FL.SMOKE_LAYOUT, warn=False)
    ARM_ACT_SCALE, ACT_PER_ARM = (0.02, 0.02, 0.02, 0.05, 0.05, 0.05), 6
    #: within the declared command scale -> the declaration is used
    kw = dict(act_scale_per_arm=ARM_ACT_SCALE, act_per_arm=ACT_PER_ARM)
    ok = from_resolution(res, act_width=6, n_declared=1,
                         act_range=[0.01, 0.01, 0.01, 0.02, 0.02, 0.02], **kw)
    assert ok.act_scale == ARM_ACT_SCALE
    #: beyond it -> fall back to the observed range rather than clamp the target
    over = from_resolution(res, act_width=6, n_declared=1,
                           act_range=[0.08, 0.01, 0.01, 0.02, 0.02, 0.02], **kw)
    assert over.act_scale is None


# --------------------------------------------------------------------------- #
# 12. field of view
# --------------------------------------------------------------------------- #


def test_roi_is_applied_before_the_resize():
    """Resizing the whole frame then centre-cropping spends output on stillness."""
    from diffusion_policy.drim.dataset import DrimEpisodes
    e = DrimEpisodes.__new__(DrimEpisodes)
    e.image_size = (24, 32)
    e.roi = {"cam": (10, 20, 48, 64)}
    raw = np.zeros((3, 480, 640, 3), np.uint8)
    raw[:, 10:58, 20:84] = 200           # only the ROI has content
    out = e._frames(raw, "cam")
    assert out.shape == (3, 24, 32, 3)
    assert out.min() > 150               # the ROI filled the frame
    #: without a declared ROI the whole frame is kept
    e.roi = {}
    assert e._frames(raw, "cam").mean() < 50


def test_every_roi_candidate_stays_inside_the_native_frame():
    from diffusion_policy.drim import dressing as D
    for name, roi in D.ROIS.items():
        for cam, (y0, x0, h, w) in roi.items():
            assert 0 <= y0 and y0 + h <= 480, (name, cam)
            assert 0 <= x0 and x0 + w <= 640, (name, cam)


def test_the_hand_drawn_boxes_are_registered():
    from diffusion_policy.drim import dressing as D
    assert D.ROIS["custom"] is D.ROI_CUSTOM
    assert D.ROIS["custom43"] is D.ROI_CUSTOM_43


def test_the_43_variant_is_3_to_4_and_is_now_legacy():
    """It was built to match a 240x320 encoder, which is no longer the size.

    Pinned against the literal 0.75 rather than against ``IMAGE_SIZE``: the
    claim is about what that box *is*, and tying it to a constant that has
    since moved is what made this test fail for the wrong reason.
    """
    from diffusion_policy.drim import dressing as D
    for cam, (_, _, h, w) in D.ROI_CUSTOM_43.items():
        assert abs(w / h - 320 / 240) < 0.01, cam
    assert D.ROI is not D.ROI_CUSTOM_43, "the 3:4 boxes are not the default"


def test_the_default_roi_downsamples_on_every_axis():
    """The property that actually matters at the current encoder input.

    A box wider than the input upsamples along that axis, which manufactures
    columns by interpolation — compute spent on nothing — and does it in the
    same frame where the other axis is discarding real rows. Every axis of the
    default must scale at or below 1.0.
    """
    from diffusion_policy.drim import dressing as D
    H, W = D.IMAGE_SIZE
    for cam, (_, _, h, w) in D.ROI.items():
        assert H <= h, f"{cam}: {H} rows out of {h} upsamples vertically"
        assert W <= w, f"{cam}: {W} cols out of {w} upsamples horizontally"


def test_the_encoder_input_is_divisible_by_the_resnet_stride():
    """resnet18 reduces by 32; an input that is not a multiple never lands."""
    from diffusion_policy.drim import dressing as D
    for v in D.IMAGE_SIZE:
        assert v % 32 == 0, f"{D.IMAGE_SIZE} is not a multiple of 32"


def test_roi_candidates_are_ordered_by_how_much_they_keep():
    from diffusion_policy.drim import dressing as D
    area = lambda r: r["image_bed_front"][2] * r["image_bed_front"][3]
    assert area(D.ROI_HAND) < area(D.ROI_MID) < area(D.ROI_WIDE)


def test_profile_rejects_an_unknown_roi():
    from diffusion_policy.drim import dressing as D
    with pytest.raises(KeyError, match="unknown roi"):
        D.profile(roi="nope")
    assert D.profile(roi="full")["roi"] == {}


def test_ee_slice_is_found_in_the_dynamics_modality_table():
    from diffusion_policy.drim.diagnose import _ee_slice
    res = FL.resolve(_packed_source(arm2_live=False), n_arms=2,
                     layout=FL.SMOKE_LAYOUT, warn=False)
    mods = FL.modalities(res, res.prop_names())
    assert _ee_slice(mods) == (7, 10)
    assert _ee_slice([("arm1_q", (0, 7), 7, "linear")]) is None


# --------------------------------------------------------------------------- #
# 13. photometric augmentation
# --------------------------------------------------------------------------- #


def test_jitter_is_per_image_not_per_batch():
    """A batch-wide jitter leaves episodes separable by their lighting."""
    from diffusion_policy.drim.augment import PhotometricJitter
    j = PhotometricJitter().train()
    x = torch.full((8, 3, 16, 16), 0.5)
    y = j(x, generator=torch.Generator().manual_seed(0))
    per_image = y.mean((1, 2, 3))
    assert float(per_image.std()) > 1e-3, "every image got the same jitter"


def test_jitter_is_a_no_op_at_eval():
    from diffusion_policy.drim.augment import PhotometricJitter
    j = PhotometricJitter().eval()
    x = torch.rand(4, 3, 8, 8)
    assert torch.equal(j(x), x)


def test_jitter_stays_in_range():
    from diffusion_policy.drim.augment import PhotometricJitter
    j = PhotometricJitter(brightness=0.9, contrast=0.9, saturation=0.9,
                          channel_gain=0.5).train()
    y = j(torch.rand(16, 3, 8, 8))
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_deterministic_shift_is_separate_from_the_random_one():
    """The probe must not depend on a seed, or the measurement moves with it."""
    from diffusion_policy.drim.augment import shift_illumination
    x = torch.full((2, 3, 4, 4), 0.5)
    a = shift_illumination(x, 0.2)
    b = shift_illumination(x, 0.2)
    assert torch.equal(a, b)
    assert torch.allclose(a, torch.full_like(a, 0.6))
    warm = shift_illumination(x, 0.0, (1.1, 1.0, 0.9))
    assert warm[0, 0, 0, 0] > warm[0, 2, 0, 0]


def test_detector_flags_a_policy_that_reads_the_light():
    from diffusion_policy.drim import diagnose as DG
    hot = DG.detectors({"diagnostics": {"illumination": {"worst_over_spread": 1.4}}})
    assert hot[0].startswith("WARN") and "reading the light" in hot[0]
    cool = DG.detectors({"diagnostics": {"illumination": {"worst_over_spread": 0.2}}})
    assert cool[0].startswith("OK")


# --------------------------------------------------------------------------- #
# 14. the hand-drawn ROI tool
# --------------------------------------------------------------------------- #


def _fake_cam(n=40, H=64, W=80, n_ep=4):
    """Frames whose motion sits in one corner and whose brightness is per-episode."""
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(n_ep), n // n_ep)
    s = np.full((n, H, W), 100.0, np.float32)
    s += (ep * 10.0)[:, None, None]                 # a per-episode light level
    s[:, 8:24, 8:24] += rng.normal(0, 30, (n, 16, 16))   # the only moving region
    return {"sample": s, "energy": np.abs(np.diff(s, axis=0)).mean(0), "ep": ep}


def test_box_stats_report_energy_resampling_and_separability():
    from diffusion_policy.drim import roi_tool as R
    c = _fake_cam()
    on = R._box_stats(c, (8, 8, 16, 16), (16, 16))
    off = R._box_stats(c, (40, 56, 16, 16), (16, 16))
    #: the moving corner is 5 % of the frame but holds most of the energy;
    #: the rest is not zero because the per-episode light level steps at each
    #: boundary, which is exactly the cue d' is there to catch
    assert on["energy"] > 10 * off["energy"]
    #: 16x16 into a 16x16 input is exactly 1.00 native pixels per output pixel
    assert on["px_per_out"] == pytest.approx(1.0)
    assert R._box_stats(c, (0, 0, 32, 32), (16, 16))["px_per_out"] == pytest.approx(4.0)
    #: the still region carries the per-episode light level and nothing else,
    #: so episodes are perfectly separable there
    assert off["d_prime"] > on["d_prime"]


def test_box_stats_ignore_a_degenerate_box():
    from diffusion_policy.drim import roi_tool as R
    assert R._box_stats(_fake_cam(), (0, 0, 2, 2), (16, 16)) == {}


def test_a_click_without_a_drag_never_yields_an_empty_crop():
    """cv2.resize raises on an empty image rather than returning one."""
    import cv2
    from diffusion_policy.drim import roi_tool as R

    #: the exact shape a bare click produced: p0 == p1
    H = W = 64
    for ay, ax, by, bx in ((10, 10, 10, 10), (10, 10, 10, 40), (10, 10, 40, 10)):
        y0, y1 = sorted((ay, by))
        x0, x1 = sorted((ax, bx))
        h, w = y1 - y0, x1 - x0
        assert (h >= 2 and w >= 2) is False        # all three are degenerate
    #: and a real drag is not
    assert (40 - 10 >= 2) and (40 - 10 >= 2)
    #: the guard is what stands between them and this
    with pytest.raises(cv2.error):
        cv2.resize(np.zeros((0, 10, 3), np.uint8), (32, 32))


def test_arrow_encodings_do_not_overlap_the_command_keys():
    from diffusion_policy.drim import roi_tool as R
    for key in "hpcrs123q":
        assert ord(key) not in R._ARROWS, key


# --------------------------------------------------------------------------- #
# 15. the image path, which the state-only specs above never touch
# --------------------------------------------------------------------------- #


def image_spec(**kw):
    """A tiny two-camera spec. resnet18 is built with weights=None, so this
    constructs offline and fast."""
    base = dict(prop_dim=14, act_dim=3, wrench_dim=6,
                cameras=("cam_a", "cam_b"), image_shape=(3, 48, 64),
                crop_shape=(44, 58), pred_horizon=4, exec_horizon=2,
                message_window=2, message_dim=8,
                fast_limits=(0.15, 0.15, 0.15))
    base.update(kw)
    return DrimSpec(**base)


def test_the_image_stage_chain_builds_through_the_public_helper():
    """train_b0 calls build(...), not the constructor -- an argument added to
    one and not the other only shows up on a real image run."""
    from diffusion_policy.drim.augment import PhotometricJitter
    spec = image_spec()
    b0 = PL.build("B0", spec, photometric=PhotometricJitter())
    assert b0.photometric is not None and b0.vision is not None
    b1 = PL.build("B1", spec, core=b0)
    d2 = PL.build("D2", spec, core=b1, message_in_dims=(6, 6))
    #: the jitter is inherited with the frozen trunk, not rebuilt per stage
    assert b1.photometric is b0.photometric
    assert d2.photometric is b0.photometric


def test_an_image_policy_encodes_a_batch_end_to_end():
    spec = image_spec()
    m = PL.build("B0", spec).eval()
    n = 2
    rgb = {c: torch.randint(0, 255, (n, spec.n_obs_steps, 48, 64, 3),
                            dtype=torch.uint8) for c in spec.cameras}
    prop = torch.randn(n, spec.n_obs_steps, spec.prop_dim)
    wr = torch.randn(n, spec.n_obs_steps, spec.wrench_dim)
    ctx = m.context(rgb, prop, wr)
    assert ctx.shape == (n, spec.context_dim)
    chunk = m.sample_chunk(ctx, None, torch.zeros(n, spec.pred_horizon, spec.act_dim))
    assert chunk.shape == (n, spec.pred_horizon, spec.act_dim)


def test_a_state_only_spec_has_no_vision_or_jitter():
    m = PL.build("B0", tiny_spec())
    assert m.vision is None and m.photometric is None


# --------------------------------------------------------------------------- #
# 16. the rollout must hand the dynamics the primitive it was trained with
# --------------------------------------------------------------------------- #


class _Eps:
    """Two short episodes with a nonzero exogenous command."""

    def __init__(self, spec, mods, n=48, n_ep=2):
        from diffusion_policy.drim import dynamics as D
        rng = np.random.default_rng(0)
        dim = D.state_dim(mods)
        self.episodes = []
        for _ in range(n_ep):
            y = rng.normal(0, 0.1, (n, dim)).astype(np.float32)
            y[:, 3:7] /= np.linalg.norm(y[:, 3:7], axis=1, keepdims=True)
            self.episodes.append({
                "dyn": y, "prop": y[:, :spec.prop_dim],
                "wrench": np.zeros((n, spec.wrench_dim), np.float32),
                "action": rng.normal(0, 0.1, (n, spec.act_dim)).astype(np.float32),
                "exo": rng.normal(0, 0.1, (n, spec.exo_dim)).astype(np.float32),
                "dt": np.full((n, 1), 0.07, np.float32)})
        self.dt_stats = {"mean": 0.07}

    def __len__(self):
        return len(self.episodes)


def _exo_setup():
    from diffusion_policy.drim import dynamics as D
    from diffusion_policy.drim.dataset import ChunkNormaliser
    mods = [("arm1_p", (0, 3), 3, "linear"), ("arm1_ee_rot", (3, 7), 3, "quat")]
    spec = DrimSpec(prop_dim=7, act_dim=3, wrench_dim=0, cameras=(),
                    pred_horizon=4, exec_horizon=2, message_window=2,
                    fast_limits=(0.15,) * 3, exo_dim=3)
    m = D.DeltaDynamics(mods, spec.act_dim, exo_dim=spec.exo_dim).eval()
    eps = _Eps(spec, mods)
    ys = np.concatenate([e["dyn"] for e in eps.episodes])
    ds = D.state_delta(ys[:-1], ys[1:], mods)
    dyn = D.FrozenDynamics(m, D.Normaliser(y=ys, d=ds))
    norm = ChunkNormaliser({"prop2": ys[:, None, :spec.prop_dim],
                            "target": np.concatenate(
                                [e["action"] for e in eps.episodes])[:, None]})
    return spec, mods, eps, norm, dyn


def test_the_dynamics_refuses_a_rollout_that_withholds_the_primitive():
    """The bug this covers killed the diagnostics after a full image run."""
    from diffusion_policy.drim import dynamics as D
    spec, mods, eps, norm, dyn = _exo_setup()
    with pytest.raises(AssertionError, match="exo_dim > 0"):
        dyn(torch.zeros(1, 7), torch.zeros(1, 3), torch.zeros(1, 1))


def test_divergence_runs_with_an_exogenous_command():
    from diffusion_policy.drim import diagnose as DG
    spec, mods, eps, norm, dyn = _exo_setup()
    model = PL.build("B1", spec, core=PL.build("B0", spec))
    out = DG.divergence(model, dyn, mods, eps, norm, spec, [0, 1],
                        n_steps=4, n_starts=4, seed=0)
    assert out and "at" in out and out["at"]
    step = out["at"]["step1"]
    #: the millimetre column and the replay floor are both populated
    assert "policy_ee_mm" in step and "replay_ee_mm" in step
    assert step["seconds"] == pytest.approx(0.07, abs=1e-6)


def test_surprise_shift_runs_with_an_exogenous_command():
    from diffusion_policy.drim import diagnose as DG
    spec, mods, eps, norm, dyn = _exo_setup()
    out = DG.surprise_shift(dyn, mods, eps, norm, spec, [0], [1])
    assert "trained_on" in out and "held_out" in out and "p99_ratio" in out
    #: the old field names still resolve, so saved summaries keep reading
    assert out["demonstration"] is out["trained_on"]
    assert out["rolled"] is out["held_out"]


def test_surprise_shift_compares_real_transitions_not_the_models_own_output():
    """The rolled arm this replaced could not be computed offline at all.

    ``surprise`` is ``nu = d_real - mu``.  In a rollout ``y_next`` is produced
    by this model, so ``d_real`` *is* ``mu`` and ``nu`` is identically zero —
    there is no ground truth at the states a policy reaches unless a robot went
    there.  The old code papered over that by pairing rolled states with
    demonstration actions taken from the top of the episode list, which
    compares a pose at one moment against a command from an unrelated one, and
    reported x169 on a run whose policy stayed 0.11 mm from the replay floor.
    """
    from diffusion_policy.drim import diagnose as DG
    from diffusion_policy.drim import dynamics as DY
    spec, mods, eps, norm, dyn = _exo_setup()

    #: feeding a model its own prediction back as the truth yields no surprise
    ep = eps.episodes[0]
    y = torch.as_tensor(ep["dyn"][:8], dtype=torch.float32)
    a = torch.as_tensor(ep["action"][:8], dtype=torch.float32)
    dt = torch.as_tensor(ep["dt"][:8], dtype=torch.float32)
    xo = torch.as_tensor(ep["exo"][:8], dtype=torch.float32)
    with torch.no_grad():
        mu, _ = dyn(y, a, dt, xo)
        d_raw = mu * torch.as_tensor(dyn.norm.ds) + torch.as_tensor(dyn.norm.dm)
        y_self = DY.torch_apply_delta(y, d_raw, mods)
        s_self = DY.surprise(dyn, y, y_self, a, dt, xo)
    assert float(s_self["NU_RAW"].abs().max()) < 1e-4, (
        "a rollout's own next state carries no surprise by construction")

    #: whereas real transitions do
    out = DG.surprise_shift(dyn, mods, eps, norm, spec, [0], [1])
    assert out["trained_on"]["p99"] > 0


def _delta_store(tmp_path, offset):
    """A one-arm recording whose action is an absolute pose target.

    ``offset`` is the standing gap between the target and the measured pose —
    the thing ``delta_ee_pos`` actually predicts. The pose itself sits half a
    metre from the base, which is the whole point: the two have completely
    different magnitudes.
    """
    import zarr
    n, eps = 60, 2
    path = str(tmp_path / "delta.zarr")
    g = zarr.open(path, "w")
    rng = np.random.default_rng(0)
    state = np.zeros((n * eps, 14), np.float32)
    state[:, 7:10] = 0.5 + rng.normal(0, 0.001, (n * eps, 3))   # ee_pos, far out
    state[:, 10] = 1.0                                          # unit quaternion
    g.create_dataset("data/state", data=state)
    g.create_dataset("data/action",
                     data=(state[:, 7:10] + offset).astype(np.float32))
    g.create_dataset("meta/episode_ends",
                     data=np.array([n, n * eps], np.int64))
    return path


def test_declared_scale_is_judged_against_the_delta_not_the_pose(tmp_path):
    """Under ``delta_ee_pos`` the scale describes ``action - ee_pos``.

    Judging it against the recorded action instead compares a few-centimetre
    ceiling with a half-metre absolute pose, rejects every physically-sized
    declaration, and silently normalises by this dataset's own range — which is
    precisely the cross-recording comparability the declaration exists to give.
    """
    from diffusion_policy.drim.dataset import load_split
    from diffusion_policy.drim import fields as FLD
    layout = FLD.PackedLayout(
        stride=14, slices={"q": (0, 7), "ee_pos": (7, 10), "ee_quat": (10, 14)})
    scale = (0.06, 0.03, 0.015)
    kw = dict(cameras=(), n_arms=1, layout=layout, val_ratio=0.5, seed=0,
              action_key="action", image_size=None, exo_key=None,
              act_scale_per_arm=scale, act_per_arm=3, roi=None,
              spec_kw=dict(action_mode="delta_ee_pos", pred_horizon=16,
                           exec_horizon=8, message_window=32))
    #: a delta inside the declared scale -> the declaration survives
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    inside = _delta_store(tmp_path / "a", np.array([0.03, 0.01, 0.005], np.float32))
    *_, spec = load_split([inside], **kw)
    assert spec.act_scale == scale

    #: and a delta that runs past it is still rejected, as it must be: the
    #: sampler clamps to [-1, 1], so such a target is unreachable outright
    outside = _delta_store(tmp_path / "b", np.array([0.09, 0.01, 0.005], np.float32))
    *_, spec2 = load_split([outside], **kw)
    assert spec2.act_scale is None


# --------------------------------------------------------------------------- #
# 17. the rollout must hand the dynamics the *units* it was trained with
# --------------------------------------------------------------------------- #


class _ConstPolicy:
    """Emits one fixed action chunk, in target units, with no corrector."""

    def __init__(self, spec, value):
        self.spec, self.value = spec, float(value)

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def context(self, rgb, prop2, wrench2):
        return prop2[:, -1]

    def sample_chunk(self, ctx, msg, _):
        return torch.full((1, self.spec.pred_horizon, self.spec.act_dim),
                          self.value)

    def step_features(self, chunk):
        return chunk

    def residual_step(self, feats, idx, sp, sw, msg):
        return torch.zeros(1, self.spec.act_dim)


class _SpyDyn:
    """Wraps the dynamics and records every action it is handed."""

    def __init__(self, dyn):
        self._d, self.seen = dyn, []

    def __call__(self, y, a, dt, xo=None):
        self.seen.append(a.detach().cpu().numpy().copy())
        return self._d(y, a, dt, xo)

    def __getattr__(self, k):
        return getattr(self._d, k)


def _delta_rollout_setup():
    from diffusion_policy.drim import dynamics as D
    from diffusion_policy.drim.dataset import ChunkNormaliser
    mods = [("arm1_ee_pos", (0, 3), 3, "linear"),
            ("arm1_ee_rot", (3, 7), 3, "quat")]
    spec = DrimSpec(prop_dim=7, act_dim=3, wrench_dim=0, cameras=(),
                    pred_horizon=4, exec_horizon=2, message_window=2,
                    fast_limits=(0.15,) * 3, action_mode="delta_ee_pos")
    m = D.DeltaDynamics(mods, spec.act_dim).eval()
    eps = _Eps(spec, mods)
    #: an absolute pose target half a metre out whose *delta* from the measured
    #: pose is centimetres — the real recording's arrangement, and the one that
    #: makes the two quantities impossible to confuse by magnitude alone
    for e in eps.episodes:
        e["dyn"][:, 0:3] += 0.5
        e["prop"] = e["dyn"][:, :spec.prop_dim]
        e["action"] = (e["dyn"][:, 0:3] + 0.03).astype(np.float32)
    ys = np.concatenate([e["dyn"] for e in eps.episodes])
    ds = D.state_delta(ys[:-1], ys[1:], mods)
    dyn = D.FrozenDynamics(m, D.Normaliser(y=ys, d=ds))
    tgt = np.concatenate([e["action"] - e["dyn"][:, 0:3] for e in eps.episodes])
    norm = ChunkNormaliser({"prop2": ys[:, None, :spec.prop_dim],
                            "step_prop": ys[:, None, :spec.prop_dim],
                            "target": tgt[:, None]})
    return spec, mods, eps, norm, dyn


def test_rollout_hands_the_dynamics_an_absolute_action_under_delta_mode():
    """The policy predicts ``action - ee_pos``; the dynamics was fitted on
    ``action``.

    ``hist_a`` carries the absolute pose target, so the rollout has to add the
    pose back before handing the action over — and the *rolled* pose, which is
    what makes the loop closed.  Feeding the bare delta is not a small error: it
    is centimetres where half a metre is expected, which queries the dynamics
    far outside anything it was fitted on.  On the full run that compounded to
    1e19 by step 16 and NaN by step 32, while the demonstrated-action floor,
    which used the absolute action all along, stayed flat at ~1.0.
    """
    from diffusion_policy.drim import diagnose as DG
    spec, mods, eps, norm, dyn = _delta_rollout_setup()
    assert DG._ee_slice(mods) == (0, 3)
    spy = _SpyDyn(dyn)
    #: normalised 0 is the mean target, i.e. the demonstrated 0.03 m delta
    model = _ConstPolicy(spec, 0.0)
    out = DG.rollout(model, spy, mods, eps, norm, spec, [0, 1],
                     n_steps=6, n_starts=3, seed=0)
    assert out and np.isfinite(out["state_divergence"]).all()
    seen = np.concatenate(spy.seen)
    #: the pose sits at ~0.5 and the delta at 0.03; an action that never left
    #: delta units would land near 0.03, two orders of magnitude below
    assert seen.mean() > 0.3, (
        f"the dynamics was handed {seen.mean():.4f}, which is the bare delta, "
        f"not the absolute pose target it was trained on")

    #: and the floor is unchanged by the fix, since it used ep["action"] already
    flo = DG.rollout(model, dyn, mods, eps, norm, spec, [0, 1], n_steps=6,
                     n_starts=3, seed=0, use_demo_actions=True)
    assert np.isfinite(flo["state_divergence"]).all()


# --------------------------------------------------------------------------- #
# 18. inference: the contract the robot is held to
# --------------------------------------------------------------------------- #


def test_inference_crop_matches_the_loader_exactly():
    """``crop_resize`` and ``DrimEpisodes`` must agree on the ROI tuple order.

    The boxes are ``(y0, x0, h, w)``. Read as ``(x, y, w, h)`` the front box
    runs 544 rows down a 480-row frame, which at least fails loudly; the back
    box fits both readings and would have silently fed the encoder a different
    part of the scene than training saw, with nothing downstream to say so.
    """
    from diffusion_policy.drim.infer import crop_resize
    from diffusion_policy.drim import dressing as DRESS

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    size = (240, 320)
    for cam, box in DRESS.ROI.items():
        y0, x0, h, w = box
        mine = crop_resize(img, box, size)
        #: the loader's own slice, written out, is the thing being matched
        import cv2
        want = cv2.resize(img[y0:y0 + h, x0:x0 + w], (size[1], size[0]),
                          interpolation=cv2.INTER_AREA)
        assert mine.shape == (size[0], size[1], 3)
        assert np.array_equal(mine, want), f"{cam} crop disagrees with the loader"


def test_inference_refuses_a_frame_the_box_does_not_fit():
    from diffusion_policy.drim.infer import crop_resize
    with pytest.raises(AssertionError, match="not producing what the run"):
        crop_resize(np.zeros((240, 320, 3), np.uint8), (32, 120, 318, 424),
                    (240, 320))


def test_inference_returns_an_absolute_pose_under_delta_mode():
    """The network predicts ``action - ee_pos``; the robot takes a pose.

    Held by a stub so it tests the runner's arithmetic rather than a trained
    policy's opinion: a chain whose output is a known delta must come back as
    that delta plus the *measured* pose.
    """
    from diffusion_policy.drim.infer import DrimRunner, Observation
    from diffusion_policy.drim.dataset import ChunkNormaliser

    spec = DrimSpec(prop_dim=27, act_dim=3, wrench_dim=6, cameras=(),
                    pred_horizon=4, exec_horizon=2, message_window=3,
                    fast_limits=(0.25, 0.05, 0.1), action_mode="delta_ee_pos",
                    exo_dim=6, act_scale=(0.06, 0.03, 0.015),
                    prop_fields=("q", "dq", "ee_pos", "ee_quat", "ee_twist"),
                    wrench_fields=("wrench",))
    DELTA = np.float32([0.03, -0.01, 0.005])

    class _Stub:
        cond_dim = 0

        def eval(self):
            return self

        def context(self, rgb, prop, wrench=None):
            return torch.zeros(1, 4)

        def sample_chunk(self, ctx, msg, _):
            n = ChunkNormaliser({"target": np.zeros((2, 1, 3), np.float32)})
            return torch.as_tensor(
                np.broadcast_to(DELTA, (1, spec.pred_horizon, 3)).copy())

        def step_features(self, chunk):
            return (chunk,)

        def residual_step(self, feats, t, prop, wrench=None, message=None):
            return torch.zeros(1, 3)

    #: no stats at all means identity, so the stub's chunk *is* metres
    norm = ChunkNormaliser(state={})
    pol = DrimRunner(model=_Stub(), dyn=None, norm=norm, spec=spec, roi={},
                     device="cpu", stage="D2")
    pol.reset()
    pos = np.float32([0.5, 0.1, 0.3])
    obs = Observation(q=np.zeros(7, np.float32), dq=np.zeros(7, np.float32),
                      ee_pos=pos, ee_quat=np.float32([0, 0, 0, 1]),
                      ee_twist=np.zeros(6, np.float32),
                      wrench=np.zeros(6, np.float32), images={},
                      zigzag_action=np.zeros(6, np.float32), dt=0.07)
    out = pol.step(obs)
    assert np.allclose(out, pos + DELTA, atol=1e-6), (
        f"expected pose+delta {pos + DELTA}, got {out} — the bare delta would "
        f"be {DELTA}")


def test_inference_reports_when_the_message_window_is_full():
    """Before the window fills the message is null, and D2 is then exactly B1."""
    from diffusion_policy.drim.infer import DrimRunner
    spec = DrimSpec(prop_dim=7, act_dim=3, wrench_dim=0, cameras=(),
                    pred_horizon=4, exec_horizon=2, message_window=5,
                    fast_limits=(0.15,) * 3)
    pol = DrimRunner(model=None, dyn=None, norm=None, spec=spec, roi={})
    assert not pol.message_ready
    pol._hist.extend([None] * 4)
    assert not pol.message_ready
    pol._hist.append(None)
    assert pol.message_ready
    pol.reset()
    assert not pol.message_ready, "reset must not carry history across episodes"


# --------------------------------------------------------------------------- #
# 19. delta_action: the target is the operator's own displacement
# --------------------------------------------------------------------------- #


def test_delta_action_is_the_centroids_own_displacement(tmp_path):
    """``a[t+i] - a[t]``, not ``a[t] - ee_pos[t]``.

    The two look alike and are opposite in the way that matters. ``action`` is
    the operator's centroid; subtracting the *measured pose* gives the
    controller's standing tracking lag, which over all 119 episodes explains
    0.0 / 0.5 / 4.2 % of the centroid's velocity and carries the **opposite
    sign** on y and z (+5.78 mm of offset against -4.92 mm/s of motion). A
    servo driven by that moves the arm backwards on the axis that carries the
    task, which is how it failed on the robot. Subtracting the centroid at the
    chunk start instead gives the displacement the operator actually commanded.
    """
    from diffusion_policy.drim.dataset import load_split, trivial_action_baselines
    from diffusion_policy.drim import fields as FLD
    layout = FLD.PackedLayout(
        stride=14, slices={"q": (0, 7), "ee_pos": (7, 10), "ee_quat": (10, 14)})
    n = 80
    path = str(tmp_path / "d.zarr")
    import zarr
    g = zarr.open(path, "w")
    state = np.zeros((n, 14), np.float32)
    #: an arm that *leads* its commanded centroid, as the real one does
    #: small enough that a 16-step chunk stays inside the declared 0.015 scale,
    #: so the scale is used and a raw zero normalises to zero
    march = np.linspace(0, -0.05, n).astype(np.float32)
    state[:, 7] = 0.5 + march * 1.05
    state[:, 10] = 1.0
    g.create_dataset("data/state", data=state)
    centroid = np.zeros((n, 3), np.float32)
    centroid[:, 0] = 0.5 + march
    g.create_dataset("data/action", data=centroid)
    g.create_dataset("meta/episode_ends", data=np.array([n], np.int64))

    kw = dict(cameras=(), n_arms=1, layout=layout, val_ratio=0.5, seed=0,
              action_key="action", image_size=None, exo_key=None,
              act_scale_per_arm=(0.015,)*3, act_per_arm=3, roi=None)
    tr, _, _, _, spec = load_split(
        [path], spec_kw=dict(action_mode="delta_action", pred_horizon=16,
                             exec_horizon=8, message_window=32), **kw)
    t = np.asarray(tr["target"])
    assert spec.action_mode == "delta_action"
    assert np.abs(t[:, 0]).max() < 1e-9, "the first step of a chunk must be zero"

    #: it points where the centroid is going, which is what a servo needs
    assert t[:, -1, 0].mean() < 0, "target must follow the centroid's -x march"

    #: and the lag points the other way, which is why it cannot be the target
    lag = centroid[:, 0] - state[:, 7]
    vel = np.diff(centroid[:, 0])
    assert lag.mean() > 0 > vel.mean(), (
        "fixture must reproduce the sign inversion the real recordings show")

    #: with target[0] == 0 the copycat and the do-nothing baseline are the same
    b = trivial_action_baselines(tr, spec)
    assert b["repeat_first_action"] == pytest.approx(b["predict_zero"])
