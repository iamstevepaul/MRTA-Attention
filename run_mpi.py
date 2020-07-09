#!/usr/bin/env python

import os
import json
import pprint as pp
import time
import torch
import torch.optim as optim
from mpi4py import MPI
# from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork
from utils import torch_load_cpu, load_problem
import mpi_opts



def run(opts, rank, comm, size):

    # Pretty print the run args
    # pp.pprint(vars(opts))
    # print(rank)
    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    # if not opts.no_tensorboard:
    #     tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    if rank == 0:
        os.makedirs(opts.save_dir)
        # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem) ##### this should be simplified

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)


    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    model = comm.bcast(model, root=0)
    model.rank = rank
    model.size = size

        # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()
    baseline.model.comm = None
    baseline = comm.bcast(baseline, root=0)
    baseline.model.rank = MPI.COMM_WORLD.Get_rank()
    baseline.model.size = MPI.COMM_WORLD.Get_size()
    # print('Rank: ', model.rank,baseline.model.rank, ' ',baseline.bl_vals)

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    if rank == 0:

    # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
        #
        # Load optimizer state
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
        #


        # Start the actual training loop
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

            torch.set_rng_state(load_data['rng_state'])
            if opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            baseline.epoch_callback(model, epoch_resume)
            print("Resuming after {}".format(epoch_resume))
            opts.epoch_start = epoch_resume + 1

    else:
        optimizer = None
        lr_scheduler = None
        val_dataset = None

    # print('Rank: ', rank ,' We hit the barrier.')
    comm.Barrier()
    #
    # model = comm.bcast(model, root=0)
    model.comm = MPI.COMM_WORLD
    # model.rank = MPI.COMM_WORLD.Get_rank()
    optimizer = comm.bcast(optimizer, root=0)
    comm.Barrier()
    # baseline = comm.bcast(baseline, root=0)
    baseline.model.comm = MPI.COMM_WORLD
    # baseline.model.rank = MPI.COMM_WORLD.Get_rank()
    # # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    val_dataset = comm.bcast(val_dataset, root=0)



    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            start_time = time.time()
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts,
                rank,
                comm
            )
            total_time = (time.time() - start_time)/60.0
            print('Epoch: ', epoch, ' time: ', total_time, ' minutes.')


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    run(get_options(), rank=rank, size=size, comm=comm)
