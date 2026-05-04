# CHANGELOG

## v0.2.0 — Speed House

### Core API
- `rp.compile(model, backend="inductor")` — torch.compile with graceful fallback
- `rp.run(domain, equation, model, strategy, devices, precision)` — master execution function
- Native Python only. No YAML, no CLI required.

### Lightning Engine
- `LightningEngine` — PyTorch Lightning backbone with manual optimization
- Adam → L-BFGS dynamic handoff (rel_std < 1e-3 triggers switch)
- `sync_dist` conditional on num_devices > 1
- Graceful fallback to native PINNTrainingRecipe when Lightning absent

### AutoScaler
- `AutoScaler.from_domain_equation(domain, equation)` — infers L0 from bounds
- Chain rule scaling for all derivative orders (dx, dxx, dt)
- Invisible to user — wired into LightningEngine.training_step

### Distribution
- `pip install rippl[distributed]` enables PyTorch Lightning
- `rp.run(..., strategy="ddp", devices=4)` for multi-GPU
- `precision="bf16-mixed"` for AMP

### Auto-Migrate
- `rp.migrate(source, framework="auto")` — AST transpiler for DeepXDE scripts
- Static analysis only — never executes foreign code
- Outputs rippl equivalent with TODO markers for manual review
- Supports: DeepXDE (full), Modulus (planned), SciANN (planned)

### Performance
- Burgers ν=0.01: loss=6.90e-11, TTS=0.1s, wall=574s
- torch.compile overhead: 0.084s (negligible)
