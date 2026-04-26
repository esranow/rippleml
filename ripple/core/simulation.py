"""ripple.core.simulation — Simulation.run() via existing FD solvers."""
from __future__ import annotations
import torch
from typing import Optional
from ripple.core.system import System
from ripple.core.solver_registry import select_solver
from ripple.physics.operators import Diffusion, Advection



class Simulation:
    """
    Drives a System forward in time using the existing FD solvers.

    Parameters
    ----------
    system  : System  (domain.spatial_dims must be 1 or 2)
    c       : wave speed
    dt, dx  : time / spatial step
    dy      : spatial step for 2-D (ignored in 1-D)
    """

    def __init__(
        self,
        system: System,
        tol: float = 1e-2,
        seed: Optional[int] = None,
    ):
        self.system = system
        self.tol = tol
        self.seed = seed

    def run(
        self,
        u0: torch.Tensor,
        v0: torch.Tensor,
        steps: int = 10,
        dt: float = 0.01,
    ) -> dict:
        self.system.set_seed(self.seed)
        self.system.validate()
        
        coords, spacings = self.system.domain.build_grid(device=u0.device)
        dx = spacings[0]
        from ripple.core.solver_registry import get_solver
        solver_fn, extractor = get_solver(self.system.equation)
        extra_kwargs = extractor(self.system.equation)
        
        if "v0" in solver_fn.__code__.co_varnames:
            res = solver_fn(u0, v0, steps=steps, dt=dt, dx=dx, **extra_kwargs)
        else:
            res = solver_fn(u0, steps=steps, dt=dt, dx=dx, **extra_kwargs)

        # Verification
        try:
            u_last = res[:, -1].detach().requires_grad_(True)
            coords_last = coords[:, -1].unsqueeze(0).expand(res.shape[0], -1, -1)
            
            pde_res = self.system.equation.compute_residual(u_last, coords_last)
            err = torch.abs(pde_res).mean().item()
            dynamic_tol = self.tol * u_last.abs().mean().clamp(min=1e-6).item()
            
            if err > dynamic_tol:
                import warnings
                warnings.warn(f"Simulation residual error {err:.4e} exceeds scaled tolerance {dynamic_tol:.4e}")
        except Exception:
            pass

        return {
            "field": res,
            "meta": {
                "dt": dt,
                "dx": dx,
                "steps": steps
            }
        }

    @staticmethod
    def visualize(
        trajectory: torch.Tensor,
        title: str = "Wave field",
        interval: int = 80,
        repeat: bool = True,
    ) -> None:
        """
        Animate a trajectory tensor.
        """
        import warnings
        t = trajectory.detach().cpu()

        # --- normalise to (T, ...) without batch/channel dims ---
        if t.dim() == 4 and t.shape[-1] == 1:   # (B,T,N,1)
            t = t[0, :, :, 0]                    # (T,N)
        elif t.dim() == 5 and t.shape[-1] == 1:  # (B,T,H,W,1)
            t = t[0, :, :, :, 0]                 # (T,H,W)
        # else assume already (T,N) or (T,H,W)

        spatial_dims = t.dim() - 1  # 1 or 2
        T = t.shape[0]

        import matplotlib.pyplot as plt
        from matplotlib import animation

        try:
            fig, ax = plt.subplots()
            vmin, vmax = t.min().item(), t.max().item()

            if spatial_dims == 1:
                line, = ax.plot(t[0].numpy())
                ax.set_ylim(vmin - 0.1 * abs(vmin), vmax + 0.1 * abs(vmax))

                def _update(frame):
                    line.set_ydata(t[frame].numpy())
                    ax.set_title(f"t = {frame}")
                    return (line,)
            else:  # 2-D heatmap
                im = ax.imshow(
                    t[0].numpy(), origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax, animated=True,
                )
                plt.colorbar(im, ax=ax)

                def _update(frame):
                    im.set_data(t[frame].numpy())
                    ax.set_title(f"t = {frame}")
                    return (im,)

            try:
                ani = animation.FuncAnimation(
                    fig, _update, frames=T,
                    interval=interval, blit=True, repeat=repeat,
                )
                plt.tight_layout()
                plt.show()
            except Exception as e:
                warnings.warn(f"Animation failed ({e}), showing static last frame.")
                _update(T - 1)
                plt.tight_layout()
                plt.show()

        except ImportError:
            # ASCII fallback — last frame, 1-D only
            frame = t[-1].flatten().tolist()
            mn, mx = min(frame), max(frame)
            span = mx - mn or 1.0
            cols = 40
            print(f"\n{title}  [last frame, ASCII]")
            for i, v in enumerate(frame[:: max(1, len(frame) // 20)]):
                bar = int((v - mn) / span * cols)
                print(f"  [{i:3d}] {'#' * bar:{cols}s} {v:.3f}")


def run_system(system: System, mode: str = "sim", **kwargs):
    """
    High-level entry point for RippleML.
    
    sim: returns trajectory tensor (B, T, ...)
    exp: returns scalar loss
    """
    if mode == "sim":
        u0 = kwargs.pop("u0", None)
        v0 = kwargs.pop("v0", None)
        if v0 is None and u0 is not None:
            v0 = torch.zeros_like(u0)
        steps = kwargs.pop("steps", 10)
        dt = kwargs.pop("dt", 0.01)
        tol = kwargs.pop("tol", 1e-2)
        return Simulation(system, tol=tol, **kwargs).run(u0, v0, steps=steps, dt=dt)
    elif mode == "exp":
        from ripple.core.experiment import Experiment
        model = kwargs.pop("model", None)
        opt = kwargs.pop("opt", None)
        x = kwargs.pop("x", None)
        t = kwargs.pop("t", None)
        return Experiment(system, model, opt).train(x, t)
    raise ValueError(f"Invalid mode: {mode}")
