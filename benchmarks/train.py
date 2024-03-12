# TODO: move dist init functions to standalone.
import argparse
import os
import torch
from functools import lru_cache
import subprocess
import random
import datetime
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from mamba_ssm.ops.selective_scan_interface import mamba_inner_ref, rearrange, repeat
import torch.nn.functional as F
from timeit import default_timer as timer
import math
from torch import nn


class MambaRef(Mamba):
    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        return mamba_inner_ref(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            ssm_ref=True,
        )


class Mampa(nn.Module):
    # Sorry i'm keeping the class here so i don't have to deal with
    def __init__(
        self,
        d_model: int,
        parallelism_size: int = 1,
        d_state: int = 16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.parallelism_size = parallelism_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        return mamba_inner_ref(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            ssm_ref=True,
        )


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0


@lru_cache()
def get_master_port(job_id: int) -> int:
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


@lru_cache()
def get_master_addr() -> str:
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    elif get_is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    else:
        return "127.0.0.1"


def init_distributed(timeout: int):
    assert isinstance(timeout, int)

    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
        timeout=datetime.timedelta(seconds=timeout),
    )

def init_torch_distributed(timeout: int = 600) -> None:
    local_rank = get_local_rank()

    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_ADDR"] = get_master_addr()
    os.environ["MASTER_PORT"] = str(
        get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1)))
    )

    if get_is_torch_run():
        print(f"Run launched with torchrun, local rank: {local_rank}")
    elif get_is_slurm_job():
        print(f"Run launched with slurm, local rank: {local_rank}")
    else:
        print("Single GPU job")

    # set GPU device
    assert 0 <= local_rank < 8
    torch.cuda.set_device(local_rank)

    # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
    # 'env://' will read these environment variables:
    # MASTER_PORT - required; has to be a free port on machine with rank 0
    # MASTER_ADDR - required (except for rank 0); address of rank 0 node
    # WORLD_SIZE - required; can be set either here, or in a call to init function
    # RANK - required; can be set either here, or in a call to init function

    # This will block until this process is called into action. This might
    # happen right away if the node is part of the initial "squad", or it
    # might be much later if it's one of the "spares" that's only needed to
    # replace another node (or it may never be needed in the end). Once that
    # happens, the process will know its "logical" rank (possibly different
    # from the "physical" rank we just got from Slurm and passed to this
    # function): only the logical rank should be used from now on.
    init_distributed(timeout=timeout)  # type: ignore

    assert get_global_rank() == torch.distributed.get_rank()
    assert get_world_size() == torch.distributed.get_world_size()

    # sanity check
    assert 0 <= local_rank <= get_global_rank() < get_world_size()


def get_data(max_iter: int, batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    # TODO: create better simulations of data.
    starting_points = torch.randint(low=1, high=vocab_size, size=(max_iter * batch_size,))

    # Generate sequences of consecutive integers
    sequences = []
    for i in range(starting_points.shape[0]):
        seq = torch.arange(starting_points[i], starting_points[i] + seq_len + 1)
        sequences.append(seq)

    # Stack the sequences into a tensor
    sequences = torch.stack(sequences)
    sequences = sequences % vocab_size
    print(f"created sequences of {sequences.shape=}")
    return sequences


def main(args) -> None:
    init_torch_distributed()
    n_layer = args.n_layer
    batch_size = args.batch_size
    max_iter = args.max_iter
    print(f"{get_global_rank()=} {args.n_layer=}")

    all_xy = get_data(max_iter=max_iter, batch_size=args.batch_size, seq_len=args.seq_len, vocab_size=args.vocab_size)
    # TODO: make actual model
    if args.model == "base":
        mamba_cls = Mamba
    elif args.model == "ref":
        mamba_cls = MambaRef
    elif args.model == "mampa":
        # TODO: specify model parallelism in this cls name.
        raise NotImplementedError("TODO")

    model = MambaLMHeadModel(
        config=MambaConfig(vocab_size=args.vocab_size, n_layer=n_layer, d_model=1024),
        device="cuda",
        dtype=torch.bfloat16,
        mamba_cls=mamba_cls,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(0, max_iter):
        tic = timer()
        xy = all_xy[step * batch_size: (step + 1) * batch_size, ...]
        x = xy[..., :-1].cuda()
        y = xy[..., 1:].cuda()
        optimizer.zero_grad()
        pred = model(x).logits
        tok_loss = F.cross_entropy(
            pred.flatten(0, 1), y.flatten(0, 1), reduction="none"
        )
        loss = tok_loss.mean()
        loss.backward()
        optimizer.step()
        tac = timer()
        torch.distributed.barrier()
        print(f"{step=} rank:{get_global_rank()} loss={float(loss.item()):.2f} iter_time: {tac - tic:.2f}s")
    pass

if __name__ == "__main__":
    # torchrun --nproc_per_node=1 benchmarks/train.py -nl 3 -m base -sl 128 -vs 512
    print("Starting")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-m", '--model', default="base", type=str, help='name of model, sent into ModelBuilder')
    parser.add_argument("-nl", '--n_layer', type=int, help='number of layers', default=8)
    parser.add_argument("-sl", '--seq_len', type=int, help='sequence length', default=8192)
    parser.add_argument("-bs", '--batch_size', type=int, help='batch size', default=4)
    parser.add_argument("-vs", '--vocab_size', type=int, help='vocab size', default=128512)
    parser.add_argument("-mi", '--max_iter', type=int, help='number of iterations', default=10)
    args = parser.parse_args()

    main(args)
