from utils.dataset import *
from os.path import exists
from os import makedirs
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch
import argparse
import datetime
import wandb
import time
import json


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Running on", device)

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

if args["model"] == "disa":
    from models.disa import *
    model = DISA(
        args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
        32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"],
        args["bilevel"], args["learned_factors"], args["scale_inv"]
    ).to(device)
    model.slot_attention.pos_emb.grid = model.slot_attention.pos_emb.grid.to(device)
    model.enc_pos_emb.grid = model.enc_pos_emb.grid.to(device)
    model.dec_pos_emb.grid = model.dec_pos_emb.grid.to(device)
elif args["model"] == "sa":
    from models.sa import *
    model = SlotAttentionAutoEncoder(
        args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
        32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"]
    ).to(device)
    model.encoder_cnn.encoder_pos.grid = model.encoder_cnn.encoder_pos.grid.to(device)
    model.decoder_cnn.decoder_pos.grid = model.decoder_cnn.decoder_pos.grid.to(device)
elif args["model"] == "isa":
    from models.isa import *
    model = ISA(
        args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
        32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"],
        args["bilevel"], args["learned_factors"], args["scale_inv"]
    ).to(device)
    model.slot_attention.pos_emb.grid = model.slot_attention.pos_emb.grid.to(device)
    model.dec_pos_emb.grid = model.dec_pos_emb.grid.to(device)
else:
    exit("Select a valid model")

print("Number of trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

if not exists(args["ckpt_path"]):
    makedirs(args["ckpt_path"])

if args["init_ckpt"] is not None:
    checkpoint = torch.load(args["ckpt_path"]+args["init_ckpt"]+".ckpt")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

criterion = nn.MSELoss()

params = [{"params": model.parameters()}]

if not args["no_wandb"]:
    wandb.init(project="DISA", entity="insert-your-username")
    wandb.run.name = args["config"]
    logs = {}
    for key, value in args.items():
        logs[key] = value
    wandb.config = logs
    wandb.watch(model)

train_set = Dataset(args["dataset"], args["data_path"], "train",
                    noise=args["noise"], crop=args["crop"], resize=args["resize"], proppred=False)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"],
                                                shuffle=True, num_workers=args["num_workers"])

optimizer = optim.Adam(params, lr=args["learning_rate"])

if args["init_ckpt"] is not None:
    optimizer.load_state_dict(checkpoint["optim_state_dict"], strict=False)

start = time.time()
if args["init_ckpt"] is not None:
    epoch, i = checkpoint["epoch"]
    epoch += 1
else:
    epoch, i = 0, 0
for epoch in range(epoch, args["num_epochs"]):
    model.train()
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

        optimizer.param_groups[0]["lr"] = learning_rate
        
        image = sample["image"].to(device)
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

    total_loss /= len(train_dataloader)

    print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, datetime.timedelta(seconds=time.time() - start)))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch": (epoch, i)
    }, args["ckpt_path"]+args["ckpt_name"]+"_"+str(epoch)+"ep.ckpt")