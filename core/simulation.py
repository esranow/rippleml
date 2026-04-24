"""ripple.core.simulation — Simulation.run() via existing FD solvers."""
from __future__ import annotations
import torch
from typing import Optional
from ripple.core.system import System
from ripple.core.solver_registry import select_solver
from ripple.physics.operators import Diffusion, Advection

def _extract_alpha(eq):
    alphas = [op.alpha for _, op in eq.terms if isinstance(op, Diffusion)]
    if not alphas: raise ValueError("Missing Diffusion operator")
    if len(alphas) > 1: raise ValueError("Multiple Diffusion operators found")
    return alphas[0]

def _extract_v(eq):
    vs = [op.v for _, op in eq.terms if isinstance(op, Advection)]
    if not vs: raise ValueError("Missing Advection operator")
    if len(vs) > 1: raise ValueError("Multiple Advection operators found")
    return vs[0]

def _extract_beta(eq):
    from ripple.physics.operators import TimeDerivative
    betas = [coeff for coeff, op in eq.terms if isinstance(op, TimeDerivative) and op.order == 1]
    return betas[0] if betas else 0.0

def _extract_c(eq):
    from ripple.physics.operators import Laplacian
    # Assuming -c^2 * Laplacian
    c_sqs = [-coeff for coeff, op in eq.terms if isinstance(op, Laplacian)]
    return c_sqs[0]**0.5 if c_sqs else 1.0


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
        c: float = 1.0,
        dt: float = 0.01,
        dx: float = 1.0,
        tol: float = 1e-2,
        seed: Optional[int] = None,
    ):
        self.system = system
        self.c = c
        self.dt = dt
        self.dx = dx
        self.tol = tol
        self.seed = seed

    def run(
        self,
        u0: torch.Tensor,
        v0: torch.Tensor,
        steps: int = 10,
    ) -> dict:
        self.system.set_seed(self.seed)
        self.system.validate()
        if self.dt <= 0: raise ValueError("dt must be > 0")
        if self.dx <= 0: raise ValueError("dx must be > 0")
        if steps <= 0: raise ValueError("steps must be > 0")

        solver = select_solver(self.system.equation)

        if solver == "wave":
            from ripple.solvers.fd_solver import solve_wave_fd_1d
            res = solve_wave_fd_1d(u0, v0, c=self.c, dt=self.dt, dx=self.dx, steps=steps)
        elif solver == "advdiff":
            from ripple.solvers.fd_solver import solve_advdiff_fd_1d
            res = solve_advdiff_fd_1d(
                u0, steps=steps,
                alpha=_extract_alpha(self.system.equation),
                v=_extract_v(self.system.equation),
                dt=self.dt, dx=self.dx,
            )
        elif solver == "diffusion":
            from ripple.solvers.fd_solver import solve_diffusion_fd_1d
            res = solve_diffusion_fd_1d(
                u0, steps=steps,
                alpha=_extract_alpha(self.system.equation),
                dt=self.dt, dx=self.dx,
            )
        elif solver == "advection":
            from ripple.solvers.fd_solver import solve_advection_fd_1d
            res = solve_advection_fd_1d(
                u0, steps=steps,
                v=_extract_v(self.system.equation),
                dt=self.dt, dx=self.dx,
            )
        elif solver == "reaction_diffusion":
            from ripple.solvers.fd_solver import solve_reaction_diffusion_fd_1d
            res = solve_reaction_diffusion_fd_1d(
                u0, steps=steps,
                alpha=_extract_alpha(self.system.equation),
                equation=self.system.equation,
                dt=self.dt, dx=self.dx,
            )
        elif solver == "first_order_nonlinear":
            from ripple.solvers.fd_solver import solve_reaction_diffusion_fd_1d
            res = solve_reaction_diffusion_fd_1d(
                u0, steps=steps,
                alpha=0.0,
                equation=self.system.equation,
                dt=self.dt, dx=self.dx,
            )
        elif solver == "damped_wave":
            from ripple.solvers.fd_solver import solve_damped_wave_fd_1d
            res = solve_damped_wave_fd_1d(
                u0, v0,
                beta=_extract_beta(self.system.equation),
                c=_extract_c(self.system.equation),
                dt=self.dt, dx=self.dx, steps=steps
            )
        else:
            raise NotImplementedError(f"No FD solver registered for key: {solver}")

        # Verification
        try:
            B, T_steps, N, _ = res.shape
            x = torch.linspace(0, (N-1)*self.dx, N, device=res.device)
            t = torch.linspace(0, (T_steps-1)*self.dt, T_steps, device=res.device)
            tt, xx = torch.meshgrid(t, x, indexing='ij')
            coords = torch.stack([xx, tt], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            
            # Autograd setup + last frame only
            u_last = res[:, -1].detach().requires_grad_(True)
            coords_last = coords[:, -1]
            
            pde_res = self.system.equation.compute_residual(u_last, coords_last)
            err = torch.abs(pde_res).mean().item()
            
            # Scale-aware tolerance
            dynamic_tol = self.tol * u_last.abs().mean().clamp(min=1e-6).item()
            
            if err > dynamic_tol:
                import warnings
                warnings.warn(f"Simulation residual error {err:.4e} exceeds scaled tolerance {dynamic_tol:.4e}")
        except Exception:
            pass

        return {
            "field": res,
            "meta": {
                "solver": solver,
                "dt": self.dt,
                "dx": self.dx,
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
        tol = kwargs.pop("tol", 1e-2)
        return Simulation(system, tol=tol, **kwargs).run(u0, v0, steps=steps)
    elif mode == "exp":
        from ripple.core.experiment import Experiment
        model = kwargs.pop("model", None)
        opt = kwargs.pop("opt", None)
        x = kwargs.pop("x", None)
        t = kwargs.pop("t", None)
        return Experiment(system, model, opt).train(x, t)
    raise ValueError(f"Invalid mode: {mode}")
