import torch


class Sampler:
    def __init__(self, net, scheduler):
        self.net = net
        self.scheduler = scheduler

    def sample(
        self,
        batch,
        conditioning_keys=None,
        uncond_tokens=None,
        num_steps=1000,
        cfg_rate=0,
        guidance_type="constant",
        guidance_start_step=0,
        guidance_end_step=None,
        generator=None,
        coherence_keys: list[str] | None = None,
        coherence_values: dict[str, float] | None = None,
        uncoherence_values: dict[str, float] | None = None,
        use_uncond_token=True,
        data_mean=None,
        data_std=None,
        sigma_data=0.5,
        sampling_dtype=None,
    ):
        x_cur = batch["y"].to(torch.float32)
        latents = batch["previous_latents"]
        if latents is not None and cfg_rate > 0:
            latents = [latents] * 2
        num_evals = 2 if cfg_rate > 0 else 1

        step_indices = torch.arange(
            num_steps + 1, dtype=torch.float32, device=x_cur.device
        )
        steps = 1 - step_indices / num_steps
        gammas = self.scheduler(steps)

        if sampling_dtype is not None:
            dtype = sampling_dtype
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if coherence_keys is not None:
            for coherence_key in coherence_keys:
                if coherence_key not in batch:
                    batch[coherence_key] = (
                        torch.ones(batch["y"].shape[0], device=batch["y"].device)
                        * coherence_values[coherence_key]
                    )

        stacked_batch = self.prepare_stacked_batch(
            batch,
            uncond_tokens,
            conditioning_keys,
            x_cur,
            coherence_keys=coherence_keys,
            coherence_values=coherence_values,
            uncoherence_values=uncoherence_values,
            use_uncond_token=use_uncond_token,
            num_evals=num_evals,
        )

        gamma_pairs = list(zip(gammas[:-1], gammas[1:]))
        for step, (gamma_now, gamma_next) in enumerate(gamma_pairs):
            last_step = step == len(gamma_pairs) - 1
            process_step_kwargs = {
                "batch": batch,
                "stacked_batch": stacked_batch,
                "cfg_rate": (
                    cfg_rate
                    if guidance_type == "constant"
                    else 2 * cfg_rate * (step / num_steps)
                ),
                "conditioning_keys": conditioning_keys,
                "step": step,
                "num_steps": num_steps,
                "guidance_start_step": guidance_start_step,
                "guidance_end_step": guidance_end_step,
            }

            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                (x_cur, latents, _gamma_prev,) = self.compute_next_step(
                    x_cur,
                    gamma_now,
                    gamma_next,
                    latents,
                    process_step_kwargs,
                    generator,
                    last_step=last_step,
                )

        return x_cur.to(torch.float32)

    def process_step(
        self,
        x_cur,
        gamma_now,
        gamma_next,
        previous_latents,
        batch,
        stacked_batch,
        cfg_rate,
        conditioning_keys,
        step,
        guidance_start_step,
        guidance_end_step,
        num_steps,
    ):
        if (
            cfg_rate > 0
            and conditioning_keys is not None
            and step >= guidance_start_step
            and (guidance_end_step is None or step <= guidance_end_step)
        ):
            return self._process_with_cfg(
                x_cur,
                gamma_now,
                previous_latents,
                stacked_batch,
                cfg_rate,
            )
        else:
            return self._process_without_guidance(
                x_cur,
                gamma_now,
                previous_latents,
                batch,
            )

    def _process_with_cfg(
        self,
        x_cur,
        gamma_now,
        previous_latents,
        stacked_batch,
        cfg_rate,
    ):
        stacked_batch["y"] = torch.cat([x_cur, x_cur], dim=0)
        stacked_batch["gamma"] = gamma_now.expand(x_cur.shape[0] * 2)
        if previous_latents is not None and previous_latents[0] is not None:
            stacked_batch["previous_latents"] = torch.cat(previous_latents, dim=0)
        else:
            stacked_batch["previous_latents"] = None
        denoised_all, latents_all = self.net(stacked_batch)
        denoised_cond, denoised_uncond = denoised_all.chunk(2, dim=0)
        if latents_all is not None:
            latents_cond, latents_uncond = latents_all.chunk(2, dim=0)
        else:
            latents_cond, latents_uncond = None, None
        denoised = denoised_cond * (1 + cfg_rate) - denoised_uncond * cfg_rate
        return denoised, [latents_cond, latents_uncond]

    def _process_without_guidance(
        self,
        x_cur,
        gamma_now,
        latents,
        batch,
    ):
        batch["y"] = x_cur
        batch["gamma"] = gamma_now.expand(x_cur.shape[0])
        batch["previous_latents"] = latents
        denoised, latents = self.net(batch)
        return denoised, latents

    def prepare_stacked_batch(
        self,
        batch,
        uncond_tokens,
        conditioning_keys,
        x_cur,
        num_evals,
        coherence_keys: list[str] | None = None,
        coherence_values: dict[str, float] | None = None,
        uncoherence_values: dict[str, float] | None = None,
        use_uncond_token: bool = True,
    ):
        stacked_batch = {}
        if uncond_tokens is None:
            use_uncond_token = False
        for key in conditioning_keys:
            if key in batch and batch[key] is None:
                stacked_batch[key] = None
                continue
            if f"{key}_mask" in batch:
                stacked_batch[f"{key}_mask"] = self._prepare_mask(
                    batch, uncond_tokens, key, use_uncond_token, num_evals
                )
            if f"{key}_embeddings" in batch:
                stacked_batch[f"{key}_embeddings"] = self._prepare_embeddings(
                    batch, uncond_tokens, key, use_uncond_token, num_evals
                )
            elif key not in batch:
                raise ValueError(f"Key {key} not in batch")
            else:
                stacked_batch[key] = self._prepare_other(
                    batch, uncond_tokens, key, use_uncond_token, num_evals
                )

        if coherence_keys:
            coherence_values = coherence_values or {}
            uncoherence_values = uncoherence_values or {}
            for key in coherence_keys:
                pos_val = coherence_values.get(key, 1.0)
                neg_val = uncoherence_values.get(key, 0.0)
                stacked_batch[key] = torch.cat(
                    [
                        torch.ones(x_cur.shape[0], device=x_cur.device) * pos_val,
                        torch.ones(x_cur.shape[0], device=x_cur.device) * neg_val,
                    ],
                    dim=0,
                ).to(x_cur.dtype)

        return stacked_batch

    def compute_next_step(self, *args, **kwargs):
        raise NotImplementedError

    def _prepare_mask(self, batch, uncond_tokens, key, use_uncond_token, num_evals):
        if use_uncond_token:
            if batch[f"{key}_mask"].shape[1] > uncond_tokens[f"{key}_mask"].shape[1]:
                uncond_mask = (
                    torch.zeros_like(batch[f"{key}_mask"])
                    if batch[f"{key}_mask"].dtype == torch.bool
                    else torch.ones_like(batch[f"{key}_mask"]) * -torch.inf
                )
                uncond_mask[:, : uncond_tokens[f"{key}_mask"].shape[1]] = uncond_tokens[
                    f"{key}_mask"
                ]
            else:
                uncond_mask = uncond_tokens[f"{key}_mask"]
                batch[f"{key}_mask"] = torch.cat(
                    [
                        batch[f"{key}_mask"],
                        torch.zeros(
                            batch[f"{key}_mask"].shape[0],
                            uncond_tokens[f"{key}_embeddings"].shape[1]
                            - batch[f"{key}_mask"].shape[1],
                            device=batch[f"{key}_mask"].device,
                            dtype=batch[f"{key}_mask"].dtype,
                        ),
                    ],
                    dim=1,
                )
            return torch.cat(
                [*[batch[f"{key}_mask"]] * (num_evals - 1), uncond_mask], dim=0
            )
        else:
            return batch[f"{key}_mask"]

    def _prepare_embeddings(
        self, batch, uncond_tokens, key, use_uncond_token, num_evals
    ):
        if use_uncond_token:
            if (
                batch[f"{key}_embeddings"].shape[1]
                > uncond_tokens[f"{key}_embeddings"].shape[1]
            ):
                uncond_tokens[f"{key}_embeddings"] = torch.cat(
                    [
                        uncond_tokens[f"{key}_embeddings"],
                        torch.zeros(
                            uncond_tokens[f"{key}_embeddings"].shape[0],
                            batch[f"{key}_embeddings"].shape[1]
                            - uncond_tokens[f"{key}_embeddings"].shape[1],
                            uncond_tokens[f"{key}_embeddings"].shape[2],
                            device=uncond_tokens[f"{key}_embeddings"].device,
                        ),
                    ],
                    dim=1,
                )
            elif (
                batch[f"{key}_embeddings"].shape[1]
                < uncond_tokens[f"{key}_embeddings"].shape[1]
            ):
                batch[f"{key}_embeddings"] = torch.cat(
                    [
                        batch[f"{key}_embeddings"],
                        torch.zeros(
                            batch[f"{key}_embeddings"].shape[0],
                            uncond_tokens[f"{key}_embeddings"].shape[1]
                            - batch[f"{key}_embeddings"].shape[1],
                            batch[f"{key}_embeddings"].shape[2],
                            device=batch[f"{key}_embeddings"].device,
                        ),
                    ],
                    dim=1,
                )
            return torch.cat(
                [
                    *[batch[f"{key}_embeddings"]] * (num_evals - 1),
                    uncond_tokens[f"{key}_embeddings"],
                ],
                dim=0,
            )
        else:
            return batch[f"{key}_embeddings"]

    def _prepare_other(self, batch, uncond_tokens, key, use_uncond_token, num_evals):
        if isinstance(batch[key], torch.Tensor):
            if use_uncond_token:
                uncond_val = uncond_tokens[key]
                if isinstance(uncond_val, (int, float)):
                    uncond_val = torch.zeros_like(batch[key])
                return torch.cat(
                    [*[batch[key]] * (num_evals - 1), uncond_val], dim=0
                )
            else:
                return batch[key]
        elif isinstance(batch[key], list):
            if use_uncond_token:
                return [
                    *[*batch[key]] * (num_evals - 1),
                    uncond_tokens[key],
                ]
            elif uncond_tokens is not None and key in uncond_tokens:
                return [
                    *[*batch[key]] * (num_evals - 1),
                    uncond_tokens[key],
                ]
            else:
                return [*[*batch[key]] * num_evals - 1, *batch[key]]
        else:
            raise ValueError("Conditioning must be a tensor or a list of tensors")


class FlowEulerSampler(Sampler):
    def compute_next_step(
        self,
        x_cur,
        gamma_now,
        gamma_next,
        previous_latents,
        process_step_kwargs,
        generator,
        last_step=False,
    ):
        denoised, previous_latents = self.process_step(
            x_cur,
            gamma_now,
            gamma_next,
            previous_latents,
            **process_step_kwargs,
        )
        x_next = x_cur - denoised * (gamma_next - gamma_now)
        return x_next, previous_latents, gamma_now


def flow_euler_sampler(net, batch, num_steps=1000, scheduler=None, **kwargs):
    sampler = FlowEulerSampler(net, scheduler)
    return sampler.sample(batch, num_steps=num_steps, **kwargs)
