from __future__ import annotations

from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor


class GibbsESSampler:
    def __init__(
        self,
        model,
        device="cpu"
    ):
        self.model = model
        self.abilities = None
        self.difficulties = None
        self.device = device

    def draw(self, n: int = 1) -> Tuple[Tensor, Tensor]:
        r"""Draw samples.

        Args:
            n: The number of samples.

        Returns:
            A `n x d`-dim tensor of `n` samples.
        """
        list_ability = []
        list_difficulty = []
        pbar = tqdm(range(n))
        for _ in pbar:
            self.step()
            list_ability.append(self.abilities.cpu())
            list_difficulty.append(self.difficulties.cpu())
            pbar.set_postfix({"llh": self.log_likelihood.item()})
        return torch.stack(list_ability), torch.stack(list_difficulty)

    def step(self) -> Tensor:
        r"""Take a step, return the new sample, update the internal state.

        Returns:
            A `d x 1`-dim sample from the domain.
        """
        if self.abilities is None:
            self.abilities, self.support_points = self.model.sample_theta_prior()
        if self.difficulties is None:
            self.difficulties = self.model.sample_item_prior()

        self.abilities = self.step_ability()
        self.difficulties = self.step_difficulty()
        return self.abilities, self.difficulties

    def step_ability(self) -> Tensor:
        r"""Take a step in the ability direction.

        Returns:
            A `d x 1`-dim sample from the domain.
        """
        nu, points = self.model.sample_theta_prior()
        theta = self._draw_angle(
            self.abilities, self.difficulties, nu=nu, is_ability=True)
        output = self._get_cart_coords(self.abilities, nu=nu, theta=theta)
        return output

    def step_difficulty(self) -> Tensor:
        r"""Take a step in the difficulty direction.

        Returns:
            A `d x 1`-dim sample from the domain.
        """
        nu = self.model.sample_item_prior()
        theta = self._draw_angle(
            self.difficulties, self.abilities, nu=nu, is_ability=False)
        output = self._get_cart_coords(self.difficulties, nu=nu, theta=theta)
        return output

    def _get_cart_coords(self, input_vec: Tensor, nu: Tensor, theta: Tensor) -> Tensor:
        r"""Determine location on ellipsoid in cartesian coordinates.

        Args:
            nu: A `d x 1`-dim tensor (the "new" direction, drawn from N(0, I)).
            theta: A `k`-dim tensor of angles.

        Returns:
            A `d x k`-dim tensor of samples from the domain in cartesian coordinates.
        """
        return input_vec * torch.cos(theta) + nu * torch.sin(theta)

    def _draw_angle(
        self,
        previous_f: Tensor,
        other_f: Tensor,
        nu: Tensor,
        is_ability: bool,
    ):
        """Draw an angle for the next sample.

        Args:
            previous_f (Tensor): The previous sample.
            other_f (Tensor): The other factor (if previous_f is ability, then other_f is difficulty).
            nu (Tensor): The new direction.
            is_ability (bool): Whether the previous sample is an ability or difficulty.

        Returns:
            Tensor: The angle.
        """
        if is_ability:
            ll_current = self.model.log_likelihood(
                ability=previous_f,
                difficulty=other_f,
                disciminatory=1,
                guessing=0,
                loading_factor=1
            )
        else:
            ll_current = self.model.log_likelihood(
                ability=other_f,
                difficulty=previous_f,
                disciminatory=1,
                guessing=0,
                loading_factor=1
            )
        ll_thres = ll_current + torch.log(torch.rand(1, device=self.device))

        angle = torch.rand(1, device=self.device) * 2 * np.pi
        angle_min, angle_max = angle - 2 * np.pi, angle

        while True:
            next_f = self._get_cart_coords(previous_f, nu, angle)
            if is_ability:
                self.log_likelihood = self.model.log_likelihood(
                    ability=next_f,
                    difficulty=other_f,
                    disciminatory=1,
                    guessing=0,
                    loading_factor=1
                )
            else:
                self.log_likelihood = self.model.log_likelihood(
                    ability=other_f,
                    difficulty=next_f,
                    disciminatory=1,
                    guessing=0,
                    loading_factor=1
                )

            if self.log_likelihood >= ll_thres:
                break
            else:
                if angle == 0:
                    break

                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = (
                    torch.rand(1, device=self.device) * (angle_max - angle_min)
                    + angle_min
                )

        return angle
