import torch

a = torch.tensor(
    [
        [[1.0, -2.0, 4.0], [4.0, 4.0, 4.0]],
        [[4.0, -2.0, 4.0], [4.0, 4.0, 4.0]],
    ]
)
print(a.shape)
r = torch.softmax(
    a,
    dim=-1,
)
print(r, r.shape)

# print(torch.stack([torch.tensor([1, 2]), torch.tensor([2, 3])]))
# print(torch.softmax(torch.randn(2, 3)))
print(torch.topk(r, k=1, dim=-1))
