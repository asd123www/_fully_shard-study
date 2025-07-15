import argparse
import os
import gzip
import json
import socket
import time
import logging
import torch
from functools import partial
from torch.autograd.profiler import record_function

from checkpoint import Checkpointer
from model import ModelArgs, Transformer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from utils import inspect_mixed_precision, inspect_model


def trace_handler(prof: torch.profiler.profile, dir_name="torch_profile_output",
                  worker_name = None, use_gzip: bool = False,
                  file_prefix="prefilling", device_id=0):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise RuntimeError("Can't create directory: " + dir_name) from e
    if not worker_name:
        worker_name = f"{socket.gethostname()}_{os.getpid()}"
    # Use nanosecond here to avoid naming clash when exporting the trace
    timestamp = time.time_ns()
    file_name = f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json"
    if use_gzip:
        file_name = file_name + ".gz"
    prof.export_chrome_trace(os.path.join(dir_name, file_name))
    # Fix the rank issue for  HolisticTraceAnalysis
    # reference: https://github.com/facebookresearch/HolisticTraceAnalysis/issues/107
    # FIXME: This does not work for json.gz
    # rn_rank = np.random.randint(low=0, high=16, dtype=int) # If there are multiple traces files, then each file should have a unique rank value.
    if use_gzip:
        with gzip.open(os.path.join(dir_name, file_name), mode="rt") as fin:
            data = json.loads(fin.read())
        data["distributedInfo"] = {"rank": device_id} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with gzip.open(os.path.join(dir_name, file_name), 'w') as fout:
            fout.write(json.dumps(data).encode('utf-8')) 
    else:
        with open(os.path.join(dir_name, file_name), "r") as fin:
            data = json.load(fin)
        data["distributedInfo"] = {"rank": device_id} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with open(os.path.join(dir_name, file_name), "w") as fout:
            json.dump(data, fout, indent=2)

    # analyzer = TraceAnalysis(trace_files={0: file_name}, trace_dir=dir_name)
    # kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(visualize=False, num_kernels=100)
    # kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    # kernel_metrics_df.to_csv(os.path.join(dir_name, f'kernel_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    # # this feature is at https://github.com/facebookresearch/HolisticTraceAnalysis/pull/209
    # # To get accurate kernel results, checkout this branch https://github.com/hychiang-git/HolisticTraceAnalysis/tree/dev/no_merge_cpu_kernels
    # if hasattr(analyzer, "get_gpu_user_annotation_breakdown"):
    #     try:
    #         user_annotation_kernel_type_metrics_df, user_annotation_metrics_df = analyzer.get_gpu_user_annotation_breakdown(visualize=False, num_kernels=100)
    #         user_annotation_kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    #         user_annotation_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    #     except Exception as e:
    #         logging.warning(f"Failed to get user annotation breakdown: {e}")
    # # Construct the memory timeline file.
    # # !!! This does not work for graph cache !!!
    # html_name = f"{file_prefix}.{worker_name}.{timestamp}.html"
    # prof.export_memory_timeline(os.path.join(dir_name, html_name), device=device)


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    torch_profile = True
    torch_profile_dir = "torch_profile_output"


    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    inspect_model(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is None:
        model.to_empty(device="cuda")
        model.reset_parameters()
    else:
        checkpointer.load_model(model)
    
    if args.mixed_precision:
        inspect_mixed_precision(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)

    # warmup
    for _ in range(5):
        with record_function("## FSDP2 forward and backward ##"):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            loss = model(x).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()

    if torch_profile:
        logging.info("Run torch profiler...")
        outfile_prefix = f"fsdp2_train"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=True, file_prefix=outfile_prefix, device_id=rank
            )
        ) as prof:
            for _ in range(10):
                with record_function("## FSDP2 forward and backward ##"):
                    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                    loss = model(x).sum()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()
                    optim.zero_grad()
                prof.step()
    else:
        for _ in range(10):
            if args.explicit_prefetching:
                model.unshard()
            
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            loss = model(x).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()

    checkpointer.save(model, optim)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
