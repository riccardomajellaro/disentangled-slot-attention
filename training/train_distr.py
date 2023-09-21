from models.disa import *
from models.sa import *
from models.isa import *
from utils.dataset import *
from tqdm import tqdm
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import torch
import argparse
import datetime
import tempfile
import wandb
import time
import json
import os


def train(gpu, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8888"

    dist.init_process_group(
        backend='nccl',
        world_size=args["world_size"],
        rank=gpu
    )
    dist.barrier()

    if args["model"] == "disa":
        model = DISA(
            args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
            32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"],
            args["bilevel"], args["learned_factors"], args["scale_inv"]
        ).to(gpu)
        model.slot_attention.pos_emb.grid = model.slot_attention.pos_emb.grid.to(gpu)
        model.enc_pos_emb.grid = model.enc_pos_emb.grid.to(gpu)
        model.dec_pos_emb.grid = model.dec_pos_emb.grid.to(gpu)
    elif args["model"] == "sa":
        model = SlotAttentionAutoEncoder(
            args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
            32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"]
        ).to(gpu)
        model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(gpu)
        model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(gpu)
    elif args["model"] == "isa":
        model = ISA(
            args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
            32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"],
            args["bilevel"], args["learned_factors"], args["scale_inv"]
        ).to(gpu)
        model.slot_attention.pos_emb.grid = model.slot_attention.pos_emb.grid.to(gpu)
        model.dec_pos_emb.grid = model.dec_pos_emb.grid.to(gpu)
    else:
        exit("Select a valid model")

    if gpu == 0:
        print("Number of trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        
    if args["init_ckpt"] is not None:
        checkpoint = torch.load(args["ckpt_path"]+args["init_ckpt"]+".ckpt")
        model.module.load_state_dict(checkpoint["model_state_dict"])
        dist.barrier()
    else:
        TMP_CHECKPOINT_PATH = tempfile.gettempdir() + f"/{args['config']}.checkpoint"
        if gpu == 0:
            torch.save(model.state_dict(), TMP_CHECKPOINT_PATH)
        dist.barrier()
        map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
        model.load_state_dict(torch.load(TMP_CHECKPOINT_PATH, map_location=map_location))

    optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"])

    if args["init_ckpt"] is not None:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    criterion = nn.MSELoss()

    if not args["no_wandb"]:
        os.environ["WANDB_RUN_GROUP"] = args["config"] + "-" + args["wandb_group_id"]
        wandb.init(project="DISA", entity="riccardomajellaro", group=args["config"])
        wandb.run.name = args["config"] + "_" + str(gpu)
        logs = {}
        for key, value in args.items():
            logs[key] = value
        wandb.config = logs
        wandb.watch(model)

    train_set = Dataset(args["dataset"], args["data_path"], "train",
                        noise=args["noise"], crop=args["crop"], resize=args["resize"], proppred=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True,
                                                                    num_replicas=args["world_size"], rank=gpu)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"],
                        num_workers=args["num_workers"], pin_memory=False, sampler=train_sampler)

    start = time.time()
    if args["init_ckpt"] is not None:
        epoch, i = checkpoint["epoch"]
        epoch += 1
    else:
        epoch, i = 0, 0
    for epoch in range(epoch, args["num_epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0

        for sample in tqdm(train_dataloader, position=0):
            i += 1

            if i < args["warmup_steps"]:
                learning_rate = args["learning_rate"] * (i / args["warmup_steps"])
                if args["slots_noise"]:
                    slots_noise = 1 - (i / args["warmup_steps"])
                else:
                    slots_noise = None
                if args["var_reg"] is not None:
                    if args["var_reg_warmup"]:
                        varreg_scale = args["var_reg"]
                    else:
                        varreg_scale = 0
            else:
                learning_rate = args["learning_rate"]
                slots_noise = None
                if args["var_reg"] is not None:
                    varreg_scale = args["var_reg"]

            learning_rate = learning_rate * (args["decay_rate"] ** (
                i / args["decay_steps"]))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            image = sample["image"].to(gpu)
            del sample
            
            if args["model"] == "sa":
                reconstruction, _, _, _, _ = model(image)
            elif args["model"] == "isa":
                reconstruction, _, _, _, _ = model(image, slots_noise=slots_noise)
            else:
                reconstruction, _, _, slots, _, _ = model(image, slots_noise=slots_noise)
            del _

            loss = criterion(reconstruction, image)
            if not args["no_wandb"]:
                wandb.log({"rec_loss": loss}, step=i)
            
            del reconstruction

            if args["var_reg"] is not None:
                var_mean_tex = slots[:, :, :args["slots_dim"]].view(-1, args["slots_dim"]).var(0).mean()
                if args["var_reg_shape"]:
                    var_mean_shape = slots[:, :, args["slots_dim"]:2*args["slots_dim"]].view(-1, args["slots_dim"]).var(0).mean()
                else:
                    var_mean_shape = 0
                loss = loss + varreg_scale * (var_mean_tex + var_mean_shape)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05, norm_type=2)
            optimizer.step()

            total_loss += loss.item()

            torch.cuda.empty_cache()
    
        total_loss /= len(train_dataloader)

        if gpu == 0:
            print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                    datetime.timedelta(seconds=time.time() - start)))

            torch.save({
                "model_state_dict": model.module.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": (epoch, i)
            }, args["ckpt_path"]+args["ckpt_name"]+"_"+str(epoch)+"ep.ckpt")
        
    if not args["no_wandb"]:
        wandb.finish()
    if gpu == 0:
        os.remove(TMP_CHECKPOINT_PATH)
    cleanup()

def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, type=str, help="Name of the configuration to load.")
    parser.add_argument("--model", default="disa", type=str, help="Name of the model to use (disa, sa, isa).")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--init_ckpt", default=None, type=str, help="Name of the checkpoint to load (without .ckpt).")
    parser.add_argument("--ckpt_path", default="checkpoints/tetrominoes/", type=str, help="Path where you want to save the model.")
    parser.add_argument("--ckpt_name", default="model", type=str, help="Name of the saved checkpoint. Set to --config if --config is not None.")
    parser.add_argument("--data_path", default="tetrominoes/", type=str, help="Path to the data.")
    parser.add_argument("--dataset", default="tetrominoes", type=str, help="Name of the dataset to use (tetrominoes, multidsprites, clevr).")
    parser.add_argument("--resolution", default=[35, 35], type=list)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--var_reg", default=0.32, help="Strength of the variance regularization term. Set to None to turn it off.")
    parser.add_argument("--var_reg_warmup", action="store_true", help="If true, use variance regularization during warmup stage.")
    parser.add_argument("--var_reg_shape", action="store_true", help="If true, use variance regularization also on shape components.")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--small_arch", action="store_true", help="Whether or not to use the small model.")
    parser.add_argument("--learned_slots", action="store_true")
    parser.add_argument("--bilevel", action="store_true")
    parser.add_argument("--slots_noise", action="store_true")
    parser.add_argument("--learned_factors", action="store_true")
    parser.add_argument("--scale_inv", action="store_true")
    parser.add_argument("--num_slots", default=4, type=int, help="Number of object slots.")
    parser.add_argument("--num_iterations", default=3, type=int, help="Number of attention iterations.")
    parser.add_argument("--slots_dim", default=32, type=int, help="Dimension of the slots.")
    parser.add_argument("--learning_rate", default=0.0004, type=float)
    parser.add_argument("--warmup_steps", default=10000, type=int, help="Number of warmup steps for the learning rate.")
    parser.add_argument("--decay_rate", default=0.5, type=float, help="Rate for the learning rate decay.")
    parser.add_argument("--decay_steps", default=100000, type=int, help="Number of steps for the learning rate decay.")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers for loading data.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument('--num_gpus', default=2, type=int, help="Number of GPUs to use.")

    args = parser.parse_args()
    args = vars(args)

    if args["config"] is not None:
        args["ckpt_name"] = args["config"]
        with open("configs/objdisc_configs.json", "r") as config_file:
            configs = json.load(config_file)[args["config"]]
        for key, value in configs.items():
            try:
                args[key] = value
            except KeyError:
                exit(f"{key} is not a valid parameter")

    args["batch_size"] = round(args["batch_size"] / args["num_gpus"])

    if not os.path.exists(args["ckpt_path"]):
        os.makedirs(args["ckpt_path"])

    if not args["no_wandb"]:
        args["wandb_group_id"] = wandb.util.generate_id()

    args["world_size"] = args["num_gpus"]
    mp.spawn(train, nprocs=args["num_gpus"], args=(args,))