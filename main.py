import torch
import os
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
import torch.nn as nn
import torch.optim as torch_optim
import math

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(32, 128, bias=False)
        self.layer2 = nn.Linear(128, 8, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


model = Model().cuda()
model = fully_shard(model, mesh=mesh)


# ### Muon Optimizer Implementation ###
# CREDIT
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
# https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Computes an approximation of the matrix zeropower using Newton-Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class FSDPMuon(torch_optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        mesh=None,
    ):
        defaults = dict(
            lr=lr, wd=wd, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps
        )
        super().__init__(params, defaults)
        self.mesh = mesh
        self.dp_group = mesh.get_group() if mesh is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.dp_group)
        self.world_size = dist.get_world_size(self.dp_group)
        # Validate that all parameters are 2D
        for p in params:
            assert p.ndim == 2, f"Muon only supports 2D parameters, got {p.ndim}"

    def adjust_lr_for_muon(self, lr, full_shape):
        A, B = full_shape
        return lr * 0.2 * math.sqrt(max(A, B))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            # Process each parameter in the group
            for p in params:
                g = p.grad
                if g is None:
                    continue  # Skip if no gradient
                if g.ndim > 2:
                    g = g.view(
                        g.size(0), -1
                    )  # Reshape higher-dimensional gradients to 2D
                assert g is not None

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                m = state["momentum_buffer"]

                g = p.grad / self.world_size
                m.mul_(momentum).add_(g)
                g_prime = g.add(m, alpha=momentum) if nesterov else m.clone()

                G_prime = g_prime.full_tensor()
                U = zeropower_via_newtonschulz5(G_prime, steps=ns_steps)

                U_replicated = DTensor.from_local(
                    U, device_mesh=p.device_mesh, placements=[Replicate()]
                )
                u = U_replicated.redistribute(placements=[Shard(0)])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

        return loss


optimizer = FSDPMuon(model.parameters(), lr=0.001, mesh=mesh)

data = torch.randn(32, 32).cuda()
target = torch.randn(32, 8).cuda()

for epoch in range(128):
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    if dist.get_rank() == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

dist.destroy_process_group()
