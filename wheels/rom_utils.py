import torch
import random
import numpy as np


class ROCTransform(torch.nn.Module):
    def __init__(self, min_step=1, min_start=1, p=1., k_vectors=16):
        super().__init__()
        self.p = p
        self.min_step = min_step
        self.min_start = min_start
        self.k_vectors = k_vectors

    def forward(self, tensor, tensora):
        """
        Args:
            tensor (Tensor): Tensor image to be transformed.

        Returns:
            Tensor: Transformed Tensor image.
        """
        if torch.rand(1) < self.p:
            U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)  # U: C H H; S: C H; Vh: C H W
            Ua, Sa, Vha = torch.linalg.svd(tensora, full_matrices=False)  # U: C H H; S: C H; Vh: C H W
            for i in range(S.shape[0]):
                # Style
                c = torch.rand(S.shape[1]).to(S.device)
                S[i] = c * S[i] + (1 - c) * Sa[i]
                self.start = self.min_start + torch.randint(low=0, high=S.shape[-1] // 2, size=(1,))[0]
                if torch.rand(1) < .5:
                    step = torch.randint(low=self.min_step, high=S.shape[-1] // 2, size=(1,))[0]
                    S[i, self.start::step] = 0
                # Content
                a = torch.distributions.Beta(1, 1).sample((self.k_vectors,)).to(U.device)
                a = a.unsqueeze(0)
                random_numbers = torch.randperm(S.shape[-1], device=S.device)[:self.k_vectors]
                random_anumbers = torch.randperm(S.shape[-1], device=S.device)[:self.k_vectors]
                U[i, :, random_numbers] = a * U[i, :, random_numbers] + (1 - a) * Ua[i, :, random_anumbers]
                b = torch.distributions.Beta(1, 1).sample((self.k_vectors,)).to(Vh.device)
                b = b.unsqueeze(1)
                random_numbers = torch.randperm(S.shape[-1], device=S.device)[:self.k_vectors]
                random_anumbers = torch.randperm(S.shape[-1], device=S.device)[:self.k_vectors]
                Vh[i, random_numbers, :] = b * Vh[i, random_numbers, :] + (1 - b) * Vha[i, random_anumbers, :]
            tensor = U @ torch.diag_embed(S) @ Vh
            tensor = torch.clamp(tensor, 0, 1)
        return tensor