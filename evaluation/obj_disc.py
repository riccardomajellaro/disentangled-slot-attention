from models.disa import *
from utils.dataset import *
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm
from torch import nn
import torch
import argparse
import json


def average_ari(masks, masks_gt, foreground_only=False):
    ari = []
    # Loop over elements in batch
    for i in range(masks.shape[0]):
        m = masks[i]
        target_masks = masks_gt[i]
        if target_masks.shape[0] < m.shape[0]:
            num_extra_target_masks = m.shape[0] - target_masks.shape[0]
            new_target_masks = target_masks[-1].repeat(num_extra_target_masks,1,1,1)
            target_masks = torch.cat((target_masks, new_target_masks), dim=0) #+ 1e-8
        m = m.cpu().numpy().flatten()
        target_masks = target_masks.cpu().numpy().flatten()
        if foreground_only:
            m = m[np.where(target_masks > 0)]
            target_masks = target_masks[np.where(target_masks > 0)]
        
        score = adjusted_rand_score(m, target_masks)
        ari.append(score)
    return sum(ari) / masks.shape[0]

def ari_score(pred_masks, target_masks):
    masks = pred_masks.detach().argmax(dim=1)
    gt_masks = target_masks.detach().argmax(dim=1)
    ari_bg = average_ari(masks, gt_masks)
    ari_fg = average_ari(masks, gt_masks, True)
    return ari_bg, ari_fg

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Running on", device)

parser = argparse.ArgumentParser()

parser.add_argument("--config", default=None, type=str, help="Name of the configuration to load.")
parser.add_argument("--model", default="disa", type=str, help="Name of the model to use (disa, sa, isa).")
parser.add_argument("--init_ckpt", default=None, type=str, help="Name of the checkpoint to load (without .ckpt).")
parser.add_argument("--ckpt_path", default="checkpoints/tetrominoes/", type=str, help="Path where the model is stored.")
parser.add_argument("--data_path", default="tetrominoes/", type=str, help="Path to the data.")
parser.add_argument("--dataset", default="tetrominoes", type=str, help="Name of the dataset to use (tetrominoes, multidsprites, clevr).")
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
parser.add_argument("--reps", default=1, type=int, help="Number of repetitions for each sample. Not needed with fixed slots.")

args = parser.parse_args()
args = vars(args)

if args["config"] is not None:
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
        False, args["learned_factors"], args["scale_inv"]
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
else:
    exit("Select a valid model")

checkpoint = torch.load(args["ckpt_path"]+args["init_ckpt"]+".ckpt")
model.load_state_dict(checkpoint["model_state_dict"])

dataset = Dataset(args["dataset"], args["data_path"], "test",
                    noise=False, crop=args["crop"], resize=args["resize"], proppred=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)

criterion = nn.MSELoss()

model.eval()
mse, ari_bg, ari_fg = 0, 0, 0
for sample in tqdm(dataloader, position=0):
    image = sample["image"].to(device)
    masks = sample["mask"].to(device)

    tmp_mse, tmp_ari_bg, tmp_ari_fg = 0, 0, 0
    for _ in range(args["reps"]):
        with torch.no_grad():
            if args["model"] == "sa":
                reconstruction, _, pred_masks, _, _ = model(image)
            else:
                reconstruction, _, pred_masks, _, _, _ = model(image)
            pred_image, _, pred_masks = model(image)[:3]

        tmp_mse += criterion(pred_image, image).item()
        bg, fg = ari_score(pred_masks.movedim(-1, 2), masks)
        tmp_ari_bg += bg 
        tmp_ari_fg += fg
    mse += tmp_mse / args["reps"]
    ari_bg += tmp_ari_bg / args["reps"]
    ari_fg += tmp_ari_fg / args["reps"]

mse /= len(dataloader)
ari_bg /= len(dataloader)
ari_fg /= len(dataloader)

with open("results/obj_disc.txt", "a") as results_f:
    res_mse = "MSE:   " + str(round(mse, 5))
    res_bg = "BG ARI:   " + str(round(ari_bg * 100, 2))
    res_fg = "FG ARI:   " + str(round(ari_fg * 100, 2))
    sep = "\n----------------------------------------------\n\n"
    results_f.write("Configs:\n")
    results_f.write("\tDataset: " + args["dataset"] + "\n")
    results_f.write("\tCheckpoint: " + args["init_ckpt"] + "\n")
    results_f.write("\tModel: " + str(args["model"]) + "\n")
    results_f.write("\n" + res_mse + "\n")
    results_f.write(res_bg + "\n")
    results_f.write(res_fg + "\n")
    results_f.write(sep)
    print("Checkpoint:", args["init_ckpt"])
    print(res_mse)
    print(res_bg)
    print(res_fg)