
import json
import os.path as osp
import glob

in_dir = "/nvme/wanghaoyu/nnUNet_root/RESULTS_FOLDER/nnUNet/3d_fullres/Task032_AMOS_CT"
res_files = glob.glob(osp.join(in_dir, "*_val*/fold_*/validation_raw/summary.json"))
for f in res_files:
    content = json.load(open(f, "r"))
    res = content['results']['mean']
    trainer_id = osp.basename(f.split("/fold")[0])
    dices = []
    for k in range(1, len(res)):
        dice = res[str(k)]['Dice']
        dices.append(dice)
    print("trainer:", trainer_id)
    print("\t".join(["%.4f"%d for d in dices])) 