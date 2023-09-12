from models.prop_pred import *
from models.disa import *
from utils.prop_pred import *
from utils.dataset import *
from tqdm import tqdm
import torch
import argparse
import json


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Running on", device)

parser = argparse.ArgumentParser()

parser.add_argument("--config", default=None, type=str, help="Name of the configuration to load.")
parser.add_argument("--inverse", action="store_true", help="True for inverse property prediction task, False for regular.")
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--disa_ckpt", default=None, type=str, help="Name of the DISA checkpoint to load (without .ckpt).")
parser.add_argument("--disa_ckpt_path", default="checkpoints/tetrominoes/", type=str, help="Path where DISA is stored.")
parser.add_argument("--proppred_ckpt", default=None, type=str, help="Name of the property prediction checkpoint to load (without .ckpt).")
parser.add_argument("--proppred_ckpt_path", default="checkpoints/tetrominoes/", type=str, help="Path where the property prediction model is stored.")
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

args = parser.parse_args()
args = vars(args)

if args["config"] is not None:
    with open("configs/proppred_configs.json", "r") as config_file:
        configs = json.load(config_file)[args["config"]]
    for key, value in configs.items():
        try:
            args[key] = value
        except KeyError:
            exit(f"{key} is not a valid parameter")

if args["dataset"] == "multidsprites":
    from torcheval.metrics import R2Score

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

checkpoint = torch.load(args["proppred_ckpt_path"]+("inv_" if args["inverse"] else "")+args["proppred_ckpt"]+".ckpt")
prop_clf.load_state_dict(checkpoint["model_state_dict"])

dataset = Dataset(args["dataset"], args["data_path"], "test",
                    noise=False, crop=args["crop"], resize=args["resize"], proppred=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)

prop_clf.eval()
encoder.eval()
reps = 1
accuracies_shapes, scores_colors, accuracies_materials = [], [], []
for _ in range(reps):
    accuracy_shapes, score_colors, accuracy_materials, tot_predictions = 0, 0, 0, 0
    for sample in tqdm(dataloader, position=0):
        pred_shapes, pred_colors, pred_materials, shapes, colors, materials = predict_properties(
                                                encoder, prop_clf, sample, args["dataset"], args["num_slots"],
                                                args["slots_dim"], args["inverse"], prop_conf, device)

        # compute and add correct predictions and r2 if needed
        tot_predictions += pred_shapes.shape[0]
        accuracy_shapes += torch.sum(shapes.argmax(dim=-1) == pred_shapes.argmax(dim=-1))
        if args["dataset"] == "multidsprites":
            r2_score = R2Score(multioutput="raw_values")
            r2_score.update(pred_colors.T, colors.T)
            score_colors += r2_score.compute().sum()
        else:
            score_colors += torch.sum(colors.argmax(dim=-1) == pred_colors.argmax(dim=-1))
        if prop_conf["material_out_dim"] is not None:
            accuracy_materials += torch.sum(materials.argmax(dim=-1) == pred_materials.argmax(dim=-1))

    # compute scores
    accuracy_shapes = accuracy_shapes / tot_predictions
    accuracies_shapes.append(accuracy_shapes.item())
    score_colors = score_colors / tot_predictions
    scores_colors.append(score_colors.item())
    if prop_conf["material_out_dim"] is not None:
        accuracy_materials = accuracy_materials / tot_predictions
        accuracies_materials.append(accuracy_materials.item())

accuracies_shapes = torch.tensor(accuracies_shapes)
scores_colors = torch.tensor(scores_colors)
accuracies_materials = torch.tensor(accuracies_materials)
with open("results/prop_pred.txt", "a") as results_f:
    res_shapes = "[Shape] Accuracy:   " + str(round(accuracies_shapes.mean().item() * 100, 2)) + " % +- " + str(round(accuracies_shapes.std().item() * 100, 2))
    if args["dataset"] == "multidsprites":
        res_colors = "[Color] R2:   " + str(round(scores_colors.mean().item(), 2)) + " +- " + str(round(scores_colors.std().item(), 2))
    else:
        res_colors = "[Color] Accuracy:   " + str(round(scores_colors.mean().item() * 100, 2)) + " % +- " + str(round(scores_colors.std().item() * 100, 2))
    if prop_conf["material_out_dim"] is not None:
        res_materials = "[Materials] Accuracy:   " + str(round(accuracies_materials.mean().item() * 100, 2)) + " % +- " + str(round(accuracies_materials.std().item() * 100, 2))
    sep = "\n----------------------------------------------\n\n"
    results_f.write("Configs:\n")
    results_f.write("\tDataset: " + args["dataset"] + "\n")
    results_f.write("\tDISA checkpoint: " + args["disa_ckpt_path"] + args["disa_ckpt"] + ".ckpt" + "\n")
    results_f.write("\tPropPred checkpoint: " + args["proppred_ckpt_path"] + ("inv_" if args["inverse"] else "") + args["proppred_ckpt"] + ".ckpt" + "\n")
    results_f.write("\tInverse: " + str(args["inverse"]) + "\n")
    results_f.write("\n" + res_shapes + "\n")
    results_f.write(res_colors + "\n")
    if prop_conf["material_out_dim"] is not None:
        results_f.write(res_materials + "\n")
    results_f.write(sep)
    print(f"[Shape] Accuracy: {accuracies_shapes.mean()} +- {accuracies_shapes.std()}")
    print(f"[Color] Accuracy: {scores_colors.mean()} +- {scores_colors.std()}")
    if prop_conf["material_out_dim"] is not None:
        print(f"[Material] Accuracy: {accuracies_materials.mean()} +- {accuracies_materials.std()}")