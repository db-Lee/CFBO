import os
import random
import argparse
import numpy as np
import json
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import wandb
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from lcpfn.bar_distribution import get_bucket_limits, BarDistribution
from lcpfn.utils import get_cosine_schedule_with_warmup, Normalize
from lcpfn.transformer import TransformerModel
from utils import InfIterator, Logger
from data.data_utils import get_dataset, HPODataset

def main(args):
    os.environ["WANDB_SILENT"] = "true"
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    if args.debug:
        args.exp_name = "debug"
        args.test_iteration = 1
        args.print_every = 1
        args.save_every = 1
        args.eval_every = 2
        budget_limit = 3
    else:
        budget_limit = 100

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dataset    
    meta_train, meta_test = get_dataset(args.data_dir, args.benchmark_name)
    if args.benchmark_name == "lcbench":
        dim_x = 7
    elif args.benchmark_name == "taskset":
        dim_x = 8
    elif args.benchmark_name == "pd1":
        dim_x = 4
    elif args.benchmark_name == "odbench":
        dim_x = 4
    else:
        raise NotImplementedError

    meta_train_sampler = HPODataset(
        meta_train,
        meta_batch_size=args.meta_batch_size,
        batch_size=args.batch_size,
        prior_batch_size=args.prior_batch_size,
        max_context=args.max_context,
        device=device,        
        meta_mixup_coeff=args.meta_mixup,
        hparam_mixup_coeff=args.hparam_mixup
    )
    meta_test_sampler = HPODataset(
        meta_test,
        meta_batch_size=args.meta_batch_size,
        batch_size=args.batch_size,
        max_context=args.max_context,
        device=device,
        meta_mixup_coeff=0.0,
        hparam_mixup_coeff=0.0
    )     

    ys = meta_train_sampler.generate_random_y(num_samples=100*args.d_output).cpu().numpy().tolist()
    ys = torch.FloatTensor(list(set(ys)))
    borders = get_bucket_limits(num_outputs=args.d_output, ys=ys, full_range=(0., 1.))
    criterion = BarDistribution(borders).to(device)

    # model and opt
    model = TransformerModel(
        dim_x=dim_x,
        d_output=args.d_output,
        d_model=args.d_model,
        dim_feedforward=2*args.d_model,
        nlayers=args.nlayers,
        dropout=args.dropout,
        data_stats=meta_train_sampler.data_stats,
        activation="gelu",
        criterion=criterion
    ).to(device)
    model.train()
    if args.use_pretrain:
        print(model.load_state_dict(torch.load("./lcpfn/pretrained_model_state_dict.pt"), strict=False))
        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    sch = get_cosine_schedule_with_warmup(opt, args.iteration//4, args.iteration)

    # logger
    logger = Logger(
        args.exp_name,
        save_dir=f"{args.save_dir}/{args.benchmark_name}/{args.exp_name}",
        save_only_last=True,
        print_every=args.print_every,
        save_every=args.save_every,
        total_step=args.iteration,
        print_to_stdout=True,
        wandb_entity=args.wandb_entity,
        wandb_project_name=args.wandb_project_name,
        wandb_config=args
    )
    logger.register_model_to_save(model, "model")
    logger.start()

    with open(f"{args.save_dir}/{args.benchmark_name}/{args.exp_name}/config_dict.json", "w") as f:
        json.dump(vars(args), f)

    if args.benchmark_name == "lcbench":
        dataset_names = ["segment", "shuttle", "sylvine", "vehicle", "volkert"]        
        num_ctx_range = [0, 2, 5, 10, 20, 30]
    elif args.benchmark_name == "taskset":
        dataset_names = ['rnn_text_classification_family_seed8', 'word_rnn_language_model_family_seed84', 'char_rnn_language_model_family_seed84']
        num_ctx_range = [0, 2, 5, 10, 20, 30]
    elif args.benchmark_name == "odbench":
        dataset_names = ["d8_a2", "d9_a2", "d10_a2"]
        num_ctx_range = [0, 1, 2, 4, 5, 10]   
    elif args.benchmark_name == "pd1":
        dataset_names = ['imagenet_resnet_batch_size_512', 'translate_wmt_xformer_translate_batch_size_64']
        num_ctx_range = [0, 2, 5, 10, 20, 30]   
    else:
        raise NotImplementedError
    
    dataset_indices = [ meta_test_sampler.dataset_names.index(dataset_name) for dataset_name in dataset_names ]
    for dataset_name in dataset_names:
        os.makedirs(f"figures/{args.benchmark_name}/{args.exp_name}/{dataset_name}", exist_ok=True)

    # outer loop
    for step in range(1, args.iteration+1):
        t_0, y_0, xc, tc, yc, xt, tt, yt = meta_train_sampler.sample()
        yt_pred = model(t_0, y_0, xc, tc, yc, xt, tt)
        losses = criterion(yt_pred, yt.squeeze(-1).contiguous())
        loss = losses.mean()
        logger.meter("meta_train", "loss", loss)

        if args.beta > 0.:
            t_0, y_0, xc, tc, yc, xt, tt, yt = meta_train_sampler.sample_prior()
            yt_pred = model(t_0, y_0, xc, tc, yc, xt, tt)
            losses = criterion(yt_pred, yt.squeeze(-1).contiguous())
            reg_loss = losses.mean()
            logger.meter("meta_train", "reg loss", reg_loss)
        else:
            reg_loss = 0.

        opt.zero_grad()
        (loss + args.beta*reg_loss).backward()
        if args.grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        opt.step()
        sch.step()

        # meta test
        if step % args.eval_every == 0 or step == args.iteration:
            # bo
            model = model.cpu()            

            output_dir = f"{args.save_dir}/{args.benchmark_name}/{args.exp_name}/bo_results"
            model_ckpt = f"{args.save_dir}/{args.benchmark_name}/{args.exp_name}/model.pt"
            config_ckpt = f"{args.save_dir}/{args.benchmark_name}/{args.exp_name}/config_dict.json"
            dataset_name = dataset_names[0]
            command = f"python bo_ours.py --output_dir {output_dir} --model_ckpt {model_ckpt} --config_ckpt {config_ckpt}"
            command += f" --benchmark_name {args.benchmark_name} --dataset_name {dataset_name} --budget_limit {budget_limit}"
            os.system(command)
            with open(f"{output_dir}/{args.benchmark_name}/{dataset_name}/normalized_regret_trajectory.json", "rb") as f:                    
                normalized_regret_trajectory = json.load(f)
            final_bo_result = normalized_regret_trajectory[-1]
            logger.meter("meta_test", dataset_name, final_bo_result)

            model = model.to(device)
            model.eval()            

            loss = 0.
            for _ in range(args.test_iteration):
                with torch.no_grad():
                    t_0, y_0, xc, tc, yc, xt, tt, yt = meta_test_sampler.sample()
                    yt_pred = model(t_0, y_0, xc, tc, yc, xt, tt)
                    losses = criterion(yt_pred, yt.squeeze(-1))
                    loss += losses.mean() / args.test_iteration
            logger.meter("meta_test", "scatter loss", loss)

            qs = [0.05, 0.5, 0.95]

            for hp_index in range(10):
                preds = []
                for num_ctx in num_ctx_range:
                    t_0, y_0, xc, tc, yc, xt, tt, yt = meta_test_sampler.sample_graph(hp_index, num_ctx)
                    with torch.no_grad():
                        logits = model(t_0, y_0, xc, tc, yc, xt, tt)
                        pred = torch.stack([ criterion.icdf(logits, q) for q in qs ], dim=-1)
                    preds.append(pred)

                x = [ _ for _ in range(1, meta_test_sampler.max_budget+1) ]
                y = yt.squeeze(-1).cpu().numpy()

                for dataset_idx, dataset_name in zip(dataset_indices, dataset_names):
                    # pred full
                    fig, axes = plt.subplots(1,6, figsize=(24, 4), sharey=True)
                    for ctx_idx, num_ctx in enumerate(num_ctx_range):
                        pred = preds[ctx_idx]

                        axes[ctx_idx].plot(x, y[dataset_idx], color='k')

                        mean = pred[dataset_idx][:, 1].cpu().numpy()
                        lower = pred[dataset_idx][:, 0].cpu().numpy()
                        upper = pred[dataset_idx][:, 2].cpu().numpy()

                        axes[ctx_idx].plot(x, mean, color='b')
                        axes[ctx_idx].fill_between(x, lower, upper, color='C0', alpha=0.5)
                        axes[ctx_idx].vlines(num_ctx, 0., 1., linewidth=0.5, color="k")

                        axes[ctx_idx].set_title(f'{num_ctx} conditioned')

                    fig.tight_layout()
                    plt.ylim(0., 1.)
                    plt.savefig(f"figures/{args.benchmark_name}/{args.exp_name}/{dataset_name}/{hp_index}.jpg")
                    plt.close()

            model.train()

        logger.step()

    logger.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=42)

    # dir
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--save_dir', type=str, default="./pretrained_surrogate_results")
    parser.add_argument('--exp_name', type=str, default="submission")

    # wandb
    parser.add_argument('--wandb_entity', type=str, default="dongbok")
    parser.add_argument('--wandb_project_name', type=str, default="CMBO-reproduce")

    # hparams for data
    parser.add_argument('--benchmark_name', type=str, default='odbench')
    parser.add_argument('--meta_batch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--prior_batch_size', type=int, default=128)
    parser.add_argument('--max_context', type=int, default=300)
    parser.add_argument('--meta_mixup', type=float, default=1.)
    parser.add_argument('--hparam_mixup', type=float, default=1.)

    # hparams for model
    parser.add_argument('--d_output', type=int, default=1000)
    parser.add_argument('--nlayers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_pretrain', action="store_true")

    # hparms for training
    parser.add_argument('--test_iteration', type=int, default=500)
    parser.add_argument('--iteration', type=int, default=100_000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)

    # hparams for logger
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--save_every', type=int, default=2000)

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    main(args)
