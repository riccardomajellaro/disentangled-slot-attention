import torch.nn.functional as F
import torch


def get_prop_configs(dataset):
    if dataset == "tetrominoes":
        return {
            "color_out_dim": 6,
            "material_out_dim": None,
            "shape_out_dim": 19 }
    elif dataset == "multidsprites":
        return {
            "color_out_dim": 3,
            "material_out_dim": None,
            "shape_out_dim": 3 }
    elif dataset == "clevr":
        return {
            "color_out_dim": 8,
            "material_out_dim": 2,
            "shape_out_dim": 3 }
    elif dataset == "clevrtex":
        return {
            "color_out_dim": None,
            "material_out_dim": 60,
            "shape_out_dim": 5 }

def to_dict(sample, clevrtex_shapes=None, clevrtex_materials=None):
    sample_ind = sample[0]
    sample = {
        "image": sample[1],
        "mask": torch.stack([torch.where(sample[2]==ind, 1., 0.) for ind in range(11)], dim=1),
    }
    if clevrtex_shapes is not None:
        sample["shape"] = clevrtex_shapes[sample_ind]
    if clevrtex_materials is not None:
        sample["material"] = clevrtex_materials[sample_ind]
    return sample

def predict_properties(encoder, prop_clf, sample, dataset, num_slots, slots_dim, inverse, prop_conf, device):
    image = sample["image"].to(device)

    # get gt masks excluding the bg (1st mask), remove ch dimension, then flatten them
    if dataset == "clevrtex":
        target_masks = sample["mask"].squeeze(2).view(image.shape[0], sample["mask"].shape[1], -1).to(device)
    else:    
        target_masks = sample["mask"][:, 1:].squeeze(2).view(image.shape[0], sample["mask"].shape[1]-1, -1).to(device)
    if dataset == "clevr":
        target_masks = target_masks[:, :6]

    # get properties and remove bg properties
    # the bg is the first element and is represented by 0
    # the -1 is to make values start from 0 instead of 1
    # on clevrtex we don't skip the bg (thus no -1)
    shapes = sample["shape"]
    if dataset != "clevrtex":
        shapes = shapes[:, 1:] - 1
    shapes = shapes.view(-1).to(device)
    if prop_conf["color_out_dim"] is not None:
        colors = sample["color"][:, 1:].to(device)
        if dataset == "tetrominoes":
            colors = torch.sum(colors * torch.tensor([1,2,4]).to(device), dim=-1)  # from binary to integer
        if dataset == "multidsprites":
            colors = colors.view(-1, 3)  # in mds the color is in RGB format
        else:
            colors -= 1
            colors = colors.view(-1)
    else:
        colors = None
    if prop_conf["material_out_dim"] is not None:
        materials = sample["material"] - 1
        if dataset != "clevrtex":
            materials = materials[:, 1:]
        materials = materials.view(-1).to(device)
    else:
        materials = None

    # predict masks and slots
    with torch.no_grad():
        _, _, masks, slots, _, _ = encoder(image)
        slots = slots[:,:,:2*slots_dim]  # exclude pos and scale embeddings
    
    # remove ch dimension to pred masks then flatten them
    masks = masks.squeeze(-1).view(image.shape[0], num_slots, -1)

    # compute cosine sim between pred and target masks, then associate the most similar
    sim_ind = torch.argmax(
        torch.einsum("bid,bjd->bij", target_masks, masks) /
        torch.einsum("bi,bj->bij", target_masks.norm(dim=-1), masks.norm(dim=-1)), dim=-1)

    # predict properties
    prop_pred = prop_clf(slots, inverse)

    # select predictions ordered by similarity indexes
    pred_shapes, pred_colors, pred_materials = [], [], []
    for j in range(image.shape[0]):
        pred_shapes.append(torch.index_select(prop_pred["shape"][j], 0, sim_ind[j]))
        if prop_conf["color_out_dim"] is not None:
            pred_colors.append(torch.index_select(prop_pred["color"][j], 0, sim_ind[j]))
        if prop_conf["material_out_dim"] is not None:
            pred_materials.append(torch.index_select(prop_pred["material"][j], 0, sim_ind[j]))
    pred_shapes = torch.stack(pred_shapes).view(-1, prop_conf["shape_out_dim"])
    if prop_conf["color_out_dim"] is not None:
        pred_colors = torch.stack(pred_colors).view(-1, prop_conf["color_out_dim"])
    if prop_conf["material_out_dim"] is not None:
        pred_materials = torch.stack(pred_materials).view(-1, prop_conf["material_out_dim"])

    # if clevr/mds, remove target and pred properties that are related to void target masks
    if dataset in ["multidsprites", "clevr", "clevrtex"]:
        nonvoid_masks_ind = (target_masks.sum(dim=-1) > 0).view(-1)
        shapes = shapes[nonvoid_masks_ind]
        pred_shapes = pred_shapes[nonvoid_masks_ind]
        if prop_conf["color_out_dim"] is not None:
            colors = colors[nonvoid_masks_ind]
            pred_colors = pred_colors[nonvoid_masks_ind]
        if prop_conf["material_out_dim"] is not None:
            materials = materials[nonvoid_masks_ind]
            pred_materials = pred_materials[nonvoid_masks_ind]

    # convert properties to one-hot vectors (except for color in mds)
    shapes = F.one_hot(shapes.type(torch.int64), num_classes=prop_conf["shape_out_dim"])
    shapes = shapes.type(torch.float32)
    if prop_conf["color_out_dim"] is not None:
        if dataset in ["tetrominoes", "clevr"]:
            colors = F.one_hot(colors.type(torch.int64), num_classes=prop_conf["color_out_dim"])
            colors = colors.type(torch.float32)
    if prop_conf["material_out_dim"] is not None:
        materials = F.one_hot(materials.type(torch.int64), num_classes=prop_conf["material_out_dim"])
        materials = materials.type(torch.float32)
    
    return pred_shapes, pred_colors, pred_materials, shapes, colors, materials