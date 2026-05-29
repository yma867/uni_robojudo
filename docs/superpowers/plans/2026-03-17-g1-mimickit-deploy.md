# G1 MimicKit Deployment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh bundle-based MimicKit deployment framework in RoboJuDo with `g1_mimickit` and `g1_mimickit_real` entrypoints, loco warmup followed by automatic MimicKit activation, a shared runtime admission gate, and an asset-replacement workflow that does not require Python edits.

**Architecture:** Introduce a new `robojudo/mimickit_deploy/` runtime that loads self-contained bundles produced by `scripts/import_mimickit_bundle.py`. Keep MimicKit-specific observation building, checkpoint parsing, motion playback, and runtime admission in that package, then expose it through a rewritten `MimicKitPolicy` and new G1 config entrypoints built on `RlLocoMimicPipeline`.

**Tech Stack:** Python, PyTorch, NumPy, PyYAML, MuJoCo/Unitree environment data, pytest/unittest, RoboJuDo config registry.

---

## File Map

- Create: `robojudo/mimickit_deploy/__init__.py`
- Create: `robojudo/mimickit_deploy/bundle.py`
- Create: `robojudo/mimickit_deploy/checkpoint.py`
- Create: `robojudo/mimickit_deploy/motion.py`
- Create: `robojudo/mimickit_deploy/char_model.py`
- Create: `robojudo/mimickit_deploy/obs_builder.py`
- Create: `robojudo/mimickit_deploy/actor.py`
- Create: `robojudo/mimickit_deploy/validator.py`
- Modify: `robojudo/policy/mimickit_policy.py`
- Modify: `robojudo/policy/policy_cfgs.py`
- Modify: `robojudo/policy/__init__.py`
- Modify: `robojudo/pipeline/rl_loco_mimic_pipeline.py`
- Create: `robojudo/config/g1/g1_mimickit_cfg.py`
- Modify: `robojudo/config/g1/__init__.py`
- Modify: `robojudo/config/g1/policy/g1_mimickit_policy_cfg.py`
- Modify: `robojudo/config/g1/g1_loco_mimic_cfg.py`
- Create: `scripts/import_mimickit_bundle.py`
- Create: `docs/mimickit_deploy.md`
- Create: `tests/test_import_mimickit_bundle.py`
- Create: `tests/test_mimickit_bundle_loader.py`
- Create: `tests/test_mimickit_checkpoint.py`
- Create: `tests/test_mimickit_obs_builder.py`
- Create: `tests/test_mimickit_validator.py`
- Create: `tests/test_g1_mimickit_config.py`
- Modify: `tests/test_full_imports.py`

## Chunk 1: Bundle And Runtime Core

### Task 1: Lock The Bundle Contract And Importer

**Files:**
- Create: `robojudo/mimickit_deploy/__init__.py`
- Create: `robojudo/mimickit_deploy/bundle.py`
- Create: `scripts/import_mimickit_bundle.py`
- Create: `tests/test_import_mimickit_bundle.py`
- Create: `tests/test_mimickit_bundle_loader.py`

- [ ] **Step 1: Write the failing importer and bundle-loader tests**

```python
def test_importer_copies_assets_and_writes_deploy_meta(tmp_path):
    ...
    assert (bundle_dir / "policy.pt").exists()
    assert meta["bundle_format_version"] == 1
    assert meta["runtime_type"] == "g1_mimickit_v1"

def test_importer_refuses_existing_bundle_without_force(tmp_path):
    ...
    assert "already exists" in str(exc.value)

def test_bundle_loader_rejects_incompatible_runtime_type(tmp_path):
    ...
    with pytest.raises(ValueError, match="runtime_type"):
        MimicKitBundle.load(bundle_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_import_mimickit_bundle.py tests/test_mimickit_bundle_loader.py -v`
Expected: FAIL because importer and loader modules do not exist yet.

- [ ] **Step 3: Implement the minimal bundle schema and importer**

```python
@dataclass(slots=True)
class MimicKitBundle:
    root: Path
    meta: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "MimicKitBundle":
        ...
        if meta["bundle_format_version"] != 1:
            raise ValueError("Unsupported bundle_format_version")
```

```python
def import_bundle(args: argparse.Namespace) -> Path:
    bundle_dir = bundle_root / args.bundle_name
    if bundle_dir.exists() and not args.force:
        raise FileExistsError(...)
    ...
    write_deploy_meta(bundle_dir / "deploy_meta.yaml", meta)
    return bundle_dir
```

- [ ] **Step 4: Re-run the importer and bundle-loader tests**

Run: `pytest tests/test_import_mimickit_bundle.py tests/test_mimickit_bundle_loader.py -v`
Expected: PASS with copy mode as default, `--force` overwrite behavior, and schema/version/type validation covered.

- [ ] **Step 5: Commit the bundle foundation**

```bash
git add robojudo/mimickit_deploy/__init__.py robojudo/mimickit_deploy/bundle.py scripts/import_mimickit_bundle.py tests/test_import_mimickit_bundle.py tests/test_mimickit_bundle_loader.py
git commit -m "feat: add mimickit bundle importer"
```

### Task 2: Parse MimicKit Checkpoints And Actor Layers Without Template Assumptions

**Files:**
- Create: `robojudo/mimickit_deploy/checkpoint.py`
- Create: `robojudo/mimickit_deploy/actor.py`
- Create: `tests/test_mimickit_checkpoint.py`
- Modify: `robojudo/mimickit_deploy/bundle.py`

- [ ] **Step 1: Write failing tests for checkpoint extraction and action de-normalization**

```python
def test_checkpoint_parser_extracts_obs_and_action_norms(tmp_path):
    ckpt = make_checkpoint_with_norm_stats(...)
    parsed = load_checkpoint(ckpt)
    assert parsed.obs_dim == 12
    assert parsed.action_dim == 6

def test_actor_forward_uses_extracted_linear_layers():
    actor = MimicKitActor.from_checkpoint(parsed)
    action = actor.forward(obs_norm)
    assert action.shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mimickit_checkpoint.py -v`
Expected: FAIL because `checkpoint.py` and `actor.py` are missing.

- [ ] **Step 3: Implement checkpoint parsing that reads actual saved tensors**

```python
@dataclass(slots=True)
class ParsedCheckpoint:
    obs_dim: int
    action_dim: int
    linear_layers: list[LinearLayer]
    obs_norm: NormalizerStats | None
    action_norm: NormalizerStats | None
```

```python
class MimicKitActor:
    def forward(self, obs_norm: np.ndarray) -> np.ndarray:
        x = obs_norm.astype(np.float32)
        for layer in self.hidden_layers:
            x = np.tanh(layer(x))
        return self.action_norm.denormalize(self.output_layer(x))
```

- [ ] **Step 4: Re-run the checkpoint tests**

Run: `pytest tests/test_mimickit_checkpoint.py -v`
Expected: PASS with dynamic layer extraction, obs normalization, and action de-normalization behavior verified.

- [ ] **Step 5: Commit the checkpoint parser**

```bash
git add robojudo/mimickit_deploy/checkpoint.py robojudo/mimickit_deploy/actor.py robojudo/mimickit_deploy/bundle.py tests/test_mimickit_checkpoint.py
git commit -m "feat: add mimickit checkpoint runtime"
```

### Task 3: Rebuild Motion Sampling, Character Ordering, And DeepMimic Observation Logic

**Files:**
- Create: `robojudo/mimickit_deploy/motion.py`
- Create: `robojudo/mimickit_deploy/char_model.py`
- Create: `robojudo/mimickit_deploy/obs_builder.py`
- Create: `tests/test_mimickit_obs_builder.py`
- Modify: `robojudo/mimickit_deploy/bundle.py`

- [ ] **Step 1: Write failing tests for motion interpolation, DFS ordering, and numeric parity**

```python
def test_motion_player_interpolates_between_neighbor_frames():
    ...
    assert np.allclose(sample.dof_pos, expected)

def test_char_model_returns_joint_and_body_dfs_order(tmp_path):
    char = MimicKitCharModel.from_xml(xml_path)
    assert char.joint_names == ["root", "left_hip", ...]

def test_obs_builder_matches_golden_reference_fixture():
    result = build_deepmimic_obs(state, motion, meta, char_model)
    assert np.allclose(result.obs_raw, fixture["obs_raw"], atol=1e-5)
    assert np.allclose(result.obs_norm, fixture["obs_norm"], atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mimickit_obs_builder.py -v`
Expected: FAIL because motion, char, and observation modules are not implemented.

- [ ] **Step 3: Implement motion playback, XML ordering, and observation construction**

```python
@dataclass(slots=True)
class MimicKitState:
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    root_lin_vel_world: np.ndarray
    root_ang_vel_world: np.ndarray
```

```python
def build_deepmimic_obs(state: MimicKitState, motion_sample: MotionSample, meta: DeployMeta, char_model: MimicKitCharModel) -> ObsBuildResult:
    ...
    return ObsBuildResult(obs_raw=obs_raw, obs_norm=obs_norm, phase=phase)
```

- [ ] **Step 4: Re-run observation tests**

Run: `pytest tests/test_mimickit_obs_builder.py -v`
Expected: PASS with at least one golden-reference fixture checking `obs_raw`, `obs_norm`, and action de-normalization parity against upstream MimicKit output.

- [ ] **Step 5: Commit the runtime core**

```bash
git add robojudo/mimickit_deploy/motion.py robojudo/mimickit_deploy/char_model.py robojudo/mimickit_deploy/obs_builder.py tests/test_mimickit_obs_builder.py
git commit -m "feat: add mimickit observation runtime"
```

## Chunk 2: Policy, Pipeline, And Delivery Surface

### Task 4: Add Validator And Rewrite MimicKitPolicy Around The New Runtime

**Files:**
- Create: `robojudo/mimickit_deploy/validator.py`
- Modify: `robojudo/policy/mimickit_policy.py`
- Modify: `robojudo/policy/policy_cfgs.py`
- Modify: `robojudo/config/g1/policy/g1_mimickit_policy_cfg.py`
- Create: `tests/test_mimickit_validator.py`

- [ ] **Step 1: Write failing tests for state admission and policy fallback behavior**

```python
def test_validator_denies_real_runtime_without_required_root_position():
    decision = validator.check_runtime(env_data_without_base_pos, runtime_mode="real")
    assert not decision.allowed
    assert "base_pos" in decision.reason

def test_policy_uses_base_state_contract_and_requests_fallback_on_motion_end():
    obs, extras = policy.get_observation(env_data, ctrl_data)
    assert extras["CALLBACK"] == ["[MOTION_DONE]"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mimickit_validator.py -v`
Expected: FAIL because validator and rewritten policy contract are not implemented.

- [ ] **Step 3: Implement validator and bundle-first policy config**

```python
class MimicKitPolicyCfg(PolicyCfg):
    bundle_name: str
    bundle_root: str | None = None
    runtime_mode: Literal["sim", "real"] = "sim"
```

```python
class MimicKitRuntimeValidator:
    def check_runtime(self, env_data, runtime_mode: str) -> AdmissionDecision:
        ...
        return AdmissionDecision(allowed=True, reason="")
```

- [ ] **Step 4: Re-run validator and policy tests**

Run: `pytest tests/test_mimickit_validator.py -v`
Expected: PASS with base-state admission checks, finite-value checks, and motion-end fallback callback behavior covered.

- [ ] **Step 5: Commit the runtime policy**

```bash
git add robojudo/mimickit_deploy/validator.py robojudo/policy/mimickit_policy.py robojudo/policy/policy_cfgs.py robojudo/config/g1/policy/g1_mimickit_policy_cfg.py tests/test_mimickit_validator.py
git commit -m "feat: rewrite mimickit policy on deploy runtime"
```

### Task 5: Add Shared Mimic Admission Gate And Register New G1 Entrypoints

**Files:**
- Create: `robojudo/config/g1/g1_mimickit_cfg.py`
- Modify: `robojudo/config/g1/__init__.py`
- Modify: `robojudo/config/g1/g1_loco_mimic_cfg.py`
- Modify: `robojudo/pipeline/rl_loco_mimic_pipeline.py`
- Modify: `tests/test_full_imports.py`
- Create: `tests/test_g1_mimickit_config.py`

- [ ] **Step 1: Write failing tests for config registration and shared gating**

```python
def test_g1_mimickit_and_real_configs_register():
    assert cfg_registry.get("g1_mimickit")
    assert cfg_registry.get("g1_mimickit_real")

def test_pipeline_routes_all_mimic_entries_through_admission_gate():
    pipeline.request_switch_to_mimic(0)
    assert pipeline._last_admission_checked == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_g1_mimickit_config.py tests/test_full_imports.py -v`
Expected: FAIL because new entrypoints and shared admission-gate plumbing are not implemented.

- [ ] **Step 3: Implement pipeline gate, fallback handling, and new entrypoints**

```python
def request_switch_to_mimic(self, mimic_idx: int) -> bool:
    decision = self.policy_manager.check_mimic_admission(mimic_idx)
    if not decision.allowed:
        logger.warning(...)
        return False
    self.policy_manager.switch_to_mimic()
    return True
```

```python
@cfg_registry.register
class g1_mimickit(G1RlLocoMimicPipelineCfg):
    loco_policy: G1LocoModePolicyCfg = G1LocoModePolicyCfg()
    mimic_policies: list[G1MimicKitPolicyCfg] = [G1MimicKitPolicyCfg(bundle_name="default", runtime_mode="sim")]
    warmup_steps: int = 100
    warmup_to_mimic: bool = True
    warmup_mimic_idx: int = 0
```

- [ ] **Step 4: Re-run config and import tests**

Run: `pytest tests/test_g1_mimickit_config.py tests/test_full_imports.py -v`
Expected: PASS with `g1_mimickit` and `g1_mimickit_real` resolvable from the registry and all mimic-activation paths forced through the same gate.

- [ ] **Step 5: Commit the integration surface**

```bash
git add robojudo/config/g1/g1_mimickit_cfg.py robojudo/config/g1/__init__.py robojudo/config/g1/g1_loco_mimic_cfg.py robojudo/pipeline/rl_loco_mimic_pipeline.py tests/test_g1_mimickit_config.py tests/test_full_imports.py
git commit -m "feat: add g1 mimickit pipeline entries"
```

### Task 6: Document The User Workflow And Verify With The Current Training Assets

**Files:**
- Create: `docs/mimickit_deploy.md`
- Modify: `docs/superpowers/specs/2026-03-17-g1-mimickit-deploy-design.md`

- [ ] **Step 1: Write the deployment doc before running manual verification**

```markdown
1. Import a new bundle with `python scripts/import_mimickit_bundle.py ...`
2. Run `python scripts/run_pipeline.py -c g1_mimickit`
3. Run `python scripts/run_pipeline.py -c g1_mimickit_real`
4. If admission fails on real hardware, remain in locomotion and inspect validator logs
```

- [ ] **Step 2: Run the importer on the user-provided assets**

Run:
`python scripts/import_mimickit_bundle.py --bundle-name default --policy MimicKit/data/models/qkf_model.pt --motion MimicKit/data/motions/hkf_mimic.pkl --log-dir MimicKit/data/logs/lo --char assets/robots/g1/mimickit/g1.xml --force`

Expected: bundle created under `assets/mimickit/g1/default/` with `deploy_meta.yaml`.

- [ ] **Step 3: Run the focused automated suite**

Run:
`pytest tests/test_import_mimickit_bundle.py tests/test_mimickit_bundle_loader.py tests/test_mimickit_checkpoint.py tests/test_mimickit_obs_builder.py tests/test_mimickit_validator.py tests/test_g1_mimickit_config.py tests/test_full_imports.py -v`

Expected: PASS.

- [ ] **Step 4: Run sim entrypoint smoke tests**

Run:
`python scripts/run_pipeline.py -c g1_mimickit`

Expected: pipeline boots, completes warmup, and only switches into MimicKit when admission passes.

- [ ] **Step 5: Run real entrypoint startup validation**

Run:
`python scripts/run_pipeline.py -c g1_mimickit_real`

Expected: startup succeeds, admission logs explain whether MimicKit can activate, and denial leaves the robot in locomotion.

- [ ] **Step 6: Commit the docs**

```bash
git add docs/mimickit_deploy.md docs/superpowers/specs/2026-03-17-g1-mimickit-deploy-design.md
git commit -m "docs: add mimickit deployment workflow"
```

## Execution Notes

- Do not reuse the existing `robojudo/mimickit_runtime` implementation or preserve its config contract.
- Use `bundle_name` as the primary v1 contract. Treat direct file paths as deprecated aliases only if they are needed to avoid import breakage.
- For v1 state admission and observation building, use only `base_pos`, `base_quat`, `base_lin_vel`, and `base_ang_vel` as the canonical root contract.
- Rotate RoboJuDo body-local `base_lin_vel` back into world coordinates before applying MimicKit global-observation logic.
- Keep all admission checks centralized. Warmup auto-switch, manual `[POLICY_MIMIC]`, and future mimic switching must all call the same gate.
- The acceptance bar is not satisfied by matching tensor shapes alone. Keep at least one golden-reference numeric parity test in the suite.

Plan complete and saved to `docs/superpowers/plans/2026-03-17-g1-mimickit-deploy.md`. Ready to execute?
