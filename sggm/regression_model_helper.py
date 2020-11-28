import torch


def generate_noise_for_model_test(x: torch.Tensor) -> torch.Tensor:
    """Generates noisy inputs to test the model out of distribution

    Args:
        x (torch.Tensor): testing inputs

    Returns:
        torch.Tensor: Noisy inputs
    """
    hypercube_min, _ = torch.min(x, dim=0)
    hypercube_max, _ = torch.max(x, dim=0)

    data_std = x.std(dim=0)
    data_mean = x.mean(dim=0)

    noise = torch.rand(x.shape).type_as(x) - 0.5
    noise *= (hypercube_max - hypercube_min) * 2 * data_std
    noise += data_mean

    return noise
