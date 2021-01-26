import torch

from functools import reduce

from sggm.types_ import List


def batch_flatten(x: torch.Tensor) -> torch.Tensor:
    # B, C, H, W -> # B, CxHxW
    return torch.flatten(x, start_dim=1)


def batch_reshape(x: torch.Tensor, shape: List[int]) -> torch.Tensor:
    # B, D -> B, C, H, W where D = CxHxW
    return x.view(-1, *shape)


def reduce_int_list(list: List[int]) -> int:
    return reduce(lambda x, y: x * y, list)


if __name__ == "__main__":
    list_test = [1, 2, 3]
    print(f"reduce_int_list: {list_test} -> {reduce_int_list(list_test)}")

    input_shape = (3, 20, 20)
    batch = torch.rand([10, *input_shape])
    batch_flattened = batch_flatten(batch)
    print(f"batch_flatten: {batch.shape} -> {batch_flattened.shape}")

    print(
        f"batch_reshape: {batch_flattened.shape} -> {batch_reshape(batch_flattened, input_shape).shape}"
    )
