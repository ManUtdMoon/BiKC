"""
Based on: https://github.com/crowsonkb/k-diffusion
Copied from consistency model
"""
import random

import numpy as np
import torch as th

from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator


def get_weightings(weight_schedule, snrs, sigma_data, **kwargs):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    elif weight_schedule == "inv_delta_sigma":
        inv_delta_sigma = kwargs.get("inv_delta_sigma", th.ones_like(snrs))
        assert inv_delta_sigma.shape == snrs.shape
        weightings = inv_delta_sigma
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l2",
        noise_schedule="uniform"
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.noise_schedule = noise_schedule
        self.rho = rho
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    ''' --- Consistency Loss --- '''
    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        local_cond=None,
        global_cond=None,
        goal=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        ## -- 生成 random noise -- ##
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        ## ----------------- 定义 denoise 调用函数 ------------------ ##

        ## 调用 denoise（model）
        def denoise_fn(x, t, local_cond, global_cond, goal):
            return self.denoise(model, x, t, local_cond, global_cond, goal)[1]

        ## 调用 denoise（target_model）
        if target_model:
            @th.no_grad()
            def target_denoise_fn(x, t, local_cond, global_cond, goal):
                return self.denoise(target_model, x, t, local_cond, global_cond, goal)[1]
        else:
            raise NotImplementedError("Must have a target model")

        ## 调用 denoise（teacher_model）
        if teacher_model:
            @th.no_grad()
            def teacher_denoise_fn(x, t, local_cond, global_cond):
                return teacher_diffusion.denoise(teacher_model, x, t, local_cond, global_cond)[1]

        ## ------------------ 定义 solver ------------------------ ##

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        ## ---------------------- 启动计算 ---------------------- ##

        ## 获取 sub-interval boundaries t_i
        if self.noise_schedule == "uniform":
            indices = th.randint(
                0, num_scales - 1, (x_start.shape[0],), device=x_start.device
            )
        elif self.noise_schedule == "lognormal":
            sigmas = get_n_sigmas_karras(
                num_scales, self.sigma_min, self.sigma_max, device=x_start.device
            )  # decreasing order
            indices = sample_lognormal_indices(
                x_start.shape[0], sigmas, device=x_start.device
            )
            assert th.all(indices <= num_scales - 1), \
                f"Invalid indices {th.max(indices)} > {num_scales - 1}"
        else:
            raise NotImplementedError(f"No such noise schedule: {self.noise_schedule}")

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        ## 获取 x_t, 即 noised trajectory || ？？ 为何要在 trajectory 上添加 noise
        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()

        ## 获取 f1_\theta(x_t)
        distiller = denoise_fn(x_t, t, local_cond, global_cond, goal)

        ## 利用 solver 获取 x_t2, 对应 Eq.6
        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)

        ## 获取 f2_\theta(x_t2), 对应 Eq.5
        distiller_target = target_denoise_fn(x_t2, t2, local_cond, global_cond, goal)
        distiller_target = distiller_target.detach()

        ## 获取 weights
        snrs = self.get_snr(t)
        assert th.all(t > t2), "sigma fed to student model must be larger"
        inv_delta_sigma = 1 / (t - t2)
        weights = get_weightings(
            self.weight_schedule, snrs, self.sigma_data,
            inv_delta_sigma=inv_delta_sigma
        )

        ## 计算 loss
        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "pseudo_huber":
            c = 0.00054 * np.sqrt(np.prod(x_start.shape[1:]))  # 0.0064, TODO 0.03/0.01
            diffs = th.sqrt((distiller - distiller_target) ** 2 + c**2) - c
            loss = mean_flat(diffs) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        term = {}
        term["loss"] = loss

        return term

    def denoise(self, model, x_t, sigmas, local_cond, global_cond, goal):
        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        if goal is None:
            model_output = model(c_in * x_t, rescaled_t, local_cond=local_cond, global_cond=global_cond)
        else:
            model_output = model(c_in * x_t, rescaled_t, local_cond=local_cond, global_cond=global_cond, goal=goal)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised


class KarrasDenoiserForTransformer(KarrasDenoiser):
    """
    Because of the different forward functions between unet and transformer,
    we need to override the denoiser (the latter only has cond, optional goal).
    """
    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        cond=None,
        goal=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        assert teacher_model is None, "Teacher model not supported for transformer"
        assert teacher_diffusion is None, "Teacher diffusion not supported for transformer"

        ## -- 生成 random noise -- ##
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        ## ----------------- 定义 denoise 调用函数 ------------------ ##

        ## 调用 denoise（model）
        def denoise_fn(x, t, cond, goal):
            return self.denoise(model, x, t, cond, goal)[1]

        ## 调用 denoise（target_model）
        if target_model:
            @th.no_grad()
            def target_denoise_fn(x, t, cond, goal):
                return self.denoise(target_model, x, t, cond, goal)[1]
        else:
            raise NotImplementedError("Must have a target model")

        ## ------------------ 定义 solver ------------------------ ##

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        ## ---------------------- 启动计算 ---------------------- ##

        ## 获取 sub-interval boundaries t_i
        if self.noise_schedule == "uniform":
            indices = th.randint(
                0, num_scales - 1, (x_start.shape[0],), device=x_start.device
            )
        elif self.noise_schedule == "lognormal":
            sigmas = get_n_sigmas_karras(
                num_scales, self.sigma_min, self.sigma_max, device=x_start.device
            )  # decreasing order
            indices = sample_lognormal_indices(
                x_start.shape[0], sigmas, device=x_start.device
            )
            assert th.all(indices <= num_scales - 1), \
                f"Invalid indices {th.max(indices)} > {num_scales - 1}"
        else:
            raise NotImplementedError(f"No such noise schedule: {self.noise_schedule}")

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        ## 获取 x_t, 即 noised trajectory || ？？ 为何要在 trajectory 上添加 noise
        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()

        ## 获取 f1_\theta(x_t)
        distiller = denoise_fn(x_t, t, cond, goal)

        ## 利用 solver 获取 x_t2, 对应 Eq.6
        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)

        ## 获取 f2_\theta(x_t2), 对应 Eq.5
        distiller_target = target_denoise_fn(x_t2, t2, cond, goal)
        distiller_target = distiller_target.detach()

        ## 获取 weights
        snrs = self.get_snr(t)
        assert th.all(t > t2), "sigma fed to student model must be larger"
        inv_delta_sigma = 1 / (t - t2)
        weights = get_weightings(
            self.weight_schedule, snrs, self.sigma_data,
            inv_delta_sigma=inv_delta_sigma
        )

        ## 计算 loss
        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "pseudo_huber":
            c = 0.00054 * np.sqrt(np.prod(x_start.shape[1:]))  # 0.0064, TODO 0.03/0.01
            diffs = th.sqrt((distiller - distiller_target) ** 2 + c**2) - c
            loss = mean_flat(diffs) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        term = {}
        term["loss"] = loss

        return term

    def denoise(self, model, x_t, sigmas, cond, goal):
        assert (not self.distillation), "Distillation not supported for transformer"

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        if goal is None:
            model_output = model(c_in * x_t, rescaled_t, cond=cond)
        else:
            model_output = model(c_in * x_t, rescaled_t, cond=cond, goal=goal)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised


def karras_sample(
    diffusion,
    model,
    shape,
    condition_data,
    condition_mask,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    local_cond=None,
    global_cond=None,
    goal=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="onestep",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    ## random input
    x_T = generator.randn(*shape, device=device) * sigma_max
    x_T[condition_mask] = condition_data[condition_mask]

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "twostep": sample_twostep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model,
                                        x_t,
                                        sigma,
                                        local_cond=local_cond,
                                        global_cond=global_cond,
                                        goal=goal)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    trajectory = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )

    # finally make sure conditioning is enforced
    trajectory[condition_mask] = condition_data[condition_mask]

    # return trajectory.clamp(-1,1)
    return trajectory


def karras_sample_transformer(
    diffusion: KarrasDenoiserForTransformer,
    model,
    shape,
    condition_data,
    condition_mask,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    cond=None,
    goal=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="onestep",
    generator=None,
):
    if generator is None:
        generator = get_generator("dummy")

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    ## random input
    x_T = generator.randn(*shape, device=device) * sigma_max
    x_T[condition_mask] = condition_data[condition_mask]

    sample_fn = {
        "onestep": sample_onestep,
        "twostep": sample_twostep,
    }[sampler]

    sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model,
                                        x_t,
                                        sigma,
                                        cond=cond,
                                        goal=goal)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    trajectory = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )

    # finally make sure conditioning is enforced
    trajectory[condition_mask] = condition_data[condition_mask]

    # return trajectory.clamp(-1,1)
    return trajectory


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_n_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """
    Constructs the noise schedule of Karras et al. (2022).
    Diffences with above: ends with sigma_min, not zero.

    @return
        sigmas: Tensor(n,)
    """
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas.to(device)    


def sample_lognormal_indices(
    num_samples: int,
    sigmas: th.Tensor,
    mean: float = -1.1,
    std: float = 2.0,
    device="cpu"
) -> th.Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw. Usually batch size
    sigmas : Tensor
        Standard deviations of the noise. len = num_scales
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.
        Each \in [0, num_scales - 1]
    """
    weights = (
        th.erf((th.log(sigmas[:-1]) - mean) / (std * np.sqrt(2))) -
        th.erf((th.log(sigmas[1:]) - mean) / (std * np.sqrt(2)))
    )
    indices = th.multinomial(weights, num_samples, replacement=True)

    return indices.to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a consistency model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)

@th.no_grad()
def sample_twostep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None
):
    """Two-step generation from a consistency model."""
    s_0 = x.new_ones([x.shape[0]])
    for i in range(2):
        x = distiller(x, sigmas[0] * s_0)
    return x


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x
