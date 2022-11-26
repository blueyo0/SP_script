
import json
import os.path as osp
import glob

in_dir = "/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/*"
res_files = glob.glob(osp.join(in_dir, "BigResUNetTrainerV4*/fold_*/validation_raw/summary.json"))
for f in res_files:
    content = json.load(open(f, "r"))
    res = content['results']['mean']
    task_id = osp.basename(osp.dirname(f.split("/fold")[0]))
    trainer_id = osp.basename(f.split("/fold")[0])
    fold_id = f.split("/fold")[-1][:6]
    dices = []
    for k in range(1, len(res)):
        dice = res[str(k)]['Dice']
        dices.append(dice)
    print("task id:", task_id)
    print("trainer:", trainer_id)
    print("trainer:", fold_id)
    print("\t".join(["%.4f"%d for d in dices])) 