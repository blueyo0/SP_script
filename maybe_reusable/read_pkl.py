
import pickle

if __name__ == "__main__":
    # path = "/mnt/cache/wanghaoyu/data/PREPROCESSED/Task033_AMOS_Task2/nnUNetData_plans_v2.1_stage1/amos_0001.pkl"
    path = "/mnt/cache/wanghaoyu/data/PREPROCESSED/Task096_Ab_tiny/nnUNetData_plans_General_stage1/Ab1K_00094.pkl"
    info = pickle.load(open(path, "rb"))
    print(info)