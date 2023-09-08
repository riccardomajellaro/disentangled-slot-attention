from models.prop_pred import *
from models.disa import *
from utils.prop_pred import *
from utils.dataset import *
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torch
import argparse
import datetime
import wandb
import time
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

parser = argparse.ArgumentParser()

parser.add_argument("--config", default=None, type=str, help="Name of the configuration to load.")
parser.add_argument("--inverse", action="store_true", help="True for inverse property prediction task, False for regular.")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--disa_ckpt", default=None, type=str, help="Name of the DISA checkpoint to load (without .ckpt).")
parser.add_argument("--disa_ckpt_path", default="checkpoints/tetrominoes/", type=str, help="Path where you want to save the model." )
parser.add_argument("--proppred_ckpt_path", default="checkpoints/tetrominoes/", type=str, help="Path where DISA is stored.")
parser.add_argument("--ckpt_name", default="model", type=str, help="Name of the saved checkpoint. Set to --config if --config is not None.")
parser.add_argument("--data_path", default="tetrominoes/", type=str, help="Path to the data.")
parser.add_argument("--resolution", default=[35, 35], type=list)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--crop", action="store_true")
parser.add_argument("--resize", action="store_true")
parser.add_argument("--small_arch", action="store_true", help="Whether or not to use the small model.")
parser.add_argument("--learned_slots", action="store_true")
parser.add_argument("--learned_factors", action="store_true")
parser.add_argument("--scale_inv", action="store_true")
parser.add_argument("--num_slots", default=4, type=int, help="Number of object slots.")
parser.add_argument("--num_iterations", default=3, type=int, help="Number of attention iterations.")
parser.add_argument("--slots_dim", default=32, type=int, help="Dimension of the slots.")
parser.add_argument("--learning_rate", default=0.0004, type=float)
parser.add_argument("--num_workers", default=0, type=int, help="Number of workers for loading data.")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs.")

args = parser.parse_args()
args = vars(args)

if args["config"] is not None:
    args["ckpt_name"] = args["config"]
    with open("configs/proppred_configs.json", "r") as config_file:
        configs = json.load(config_file)[args["config"]]
    for key, value in configs.items():
        try:
            args[key] = value
        except KeyError:
            exit(f"{key} is not a valid parameter")

encoder = DISA(
    args["resolution"], args["num_slots"], args["num_iterations"], args["slots_dim"],
    32 if args["small_arch"] else 64, args["small_arch"], args["learned_slots"],
    False, args["learned_factors"], args["scale_inv"]
).to(device)
encoder.slot_attention.pos_emb.grid = encoder.slot_attention.pos_emb.grid.to(device)
encoder.enc_pos_emb.grid = encoder.enc_pos_emb.grid.to(device)
encoder.dec_pos_emb.grid = encoder.dec_pos_emb.grid.to(device)

checkpoint = torch.load(args["disa_ckpt_path"]+args["disa_ckpt"]+".ckpt")
encoder.load_state_dict(checkpoint["model_state_dict"])

prop_conf = get_prop_configs(args["dataset"])

prop_clf = PropertyClf(color_out_dim=prop_conf["color_out_dim"],
                       material_out_dim=prop_conf["material_out_dim"],
                       shape_out_dim=prop_conf["shape_out_dim"]).to(device)

params = [{"params": prop_clf.parameters()}]

criterion_ce = nn.CrossEntropyLoss()
if args["dataset"] == "multidsprites":
    criterion_mse = nn.MSELoss()

if not args["no_wandb"]:
    wandb.init(project="DISA", entity="riccardomajellaro")
    wandb.run.name = "proppred_" + args["config"]
    if args["inverse"]:
        wandb.run.name = "inv_" + wandb.run.name
    logs = {}
    for key, value in args.items():
        logs[key] = value
    wandb.config = logs
    wandb.watch(prop_clf)

optimizer = optim.Adam(params, lr=args["learning_rate"])

# load train set
dataset = Dataset(args["dataset"], args["data_path"], "train",
                    noise=False, crop=args["crop"], resize=args["resize"], proppred=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])

# load val set
dataset_val = Dataset(args["dataset"], args["data_path"], "val",
                    noise=False, crop=args["crop"], resize=args["resize"], proppred=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"])

start = time.time()
epoch, i = 0, 0
prop_clf.train()
encoder.eval()
for epoch in range(args["num_epochs"]):
    total_color_loss, total_shape_loss = 0, 0
    if prop_conf["material_out_dim"] is not None:
        total_material_loss = 0

    for sample in tqdm(dataloader, position=0):
        i += 1

        pred_shapes, pred_colors, pred_materials, shapes, colors, materials = predict_properties(
                                                encoder, prop_clf, sample, args["dataset"], args["num_slots"],
                                                args["slots_dim"], args["inverse"], prop_conf, device)

        # compute losses
        shape_loss = criterion_ce(pred_shapes, shapes)
        if args["dataset"] == "multidsprites":
            color_loss = criterion_mse(pred_colors, colors)
        else:
            color_loss = criterion_ce(pred_colors, colors)
        loss = shape_loss + color_loss
        if prop_conf["material_out_dim"] is not None:
            material_loss = criterion_ce(pred_materials, materials)
            loss += material_loss
        if not args["no_wandb"]:
            loss_dict = {"shape_loss": shape_loss, "color_loss": color_loss}
            if prop_conf["material_out_dim"] is not None:
                loss_dict["material_loss"] = material_loss
            wandb.log(loss_dict, step=i)
        
        total_shape_loss += shape_loss.item()
        total_color_loss += color_loss.item()
        if prop_conf["material_out_dim"] is not None:
            total_material_loss += material_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i-1) % 250 == 0:
            prop_clf.eval()
            val_loss = [0, 0, 0]
            for sample in dataloader_val:
                with torch.no_grad():
                    pred_shapes, pred_colors, pred_materials, shapes, colors, materials = predict_properties(
                                                            encoder, prop_clf, sample, args["dataset"], args["num_slots"],
                                                            args["slots_dim"], args["inverse"], prop_conf, device)
                # compute validation losses
                shape_loss = criterion_ce(pred_shapes, shapes)
                val_loss[0] += shape_loss
                if args["dataset"] == "multidsprites":
                    color_loss = criterion_mse(pred_colors, colors)
                else:
                    color_loss = criterion_ce(pred_colors, colors)
                val_loss[1] += color_loss
                if prop_conf["material_out_dim"] is not None:
                    material_loss = criterion_ce(pred_materials, materials)
                    val_loss[2] += material_loss
            val_loss[0] /= len(dataloader_val)
            val_loss[1] /= len(dataloader_val)
            val_loss[2] /= len(dataloader_val)
            if not args["no_wandb"]:
                loss_dict = {"shape_loss_val": val_loss[0], "color_loss_val": val_loss[1]}
                if prop_conf["material_out_dim"] is not None:
                    loss_dict["material_loss_val"] = val_loss[2]
                wandb.log(loss_dict, step=i) 
            prop_clf.train()
    
    total_shape_loss /= len(dataloader)
    total_color_loss /= len(dataloader)
    if prop_conf["material_out_dim"] is not None:
        total_material_loss /= len(dataloader)

    print(f"Epoch: {epoch}, Shape Loss: {total_shape_loss}, Color Loss: {total_color_loss},"
        + f"Material Loss: {total_material_loss}" if prop_conf["material_out_dim"] is not None else "" \
        + f"Time: {datetime.timedelta(seconds=time.time() - start)}")

    if epoch % 5 == 0:
        torch.save({
            "model_state_dict": prop_clf.state_dict(),
            }, args["proppred_ckpt_path"] + ("inv_" if args["inverse"] else "") + "proppred_" + args["ckpt_name"]+"_"+str(epoch)+"ep.ckpt")

torch.save({
    "model_state_dict": prop_clf.state_dict(),
}, args["proppred_ckpt_path"] + ("inv_" if args["inverse"] else "") + "proppred_" + args["ckpt_name"]+"_"+str(epoch)+"ep.ckpt")