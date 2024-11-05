from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from utils import set_seed

set_seed(42)
all_ds = {}
for cls in [
    "CC01",
    "CC03",
    "CN02",
    "DT01",
    "L02",
    "L04",
    "L06",
    "L07",
    "L08",
    "L09",
    "train",
]:
    # for cls in ["CC01", "CN02", "train"]:
    ds = load_dataset("stair-lab/dsa_hk231_wtc_per_student_sft_lf", split=cls)
    new_student_idxs = []
    for idx, hist in enumerate(ds["history"]):
        if len(hist) == 0:
            new_student_idxs.append(idx)

    total_students = len(new_student_idxs)
    start_test_idx = new_student_idxs[int(total_students * 0.8)]
    new_ds = ds.select(list(range(0, start_test_idx)))
    new_ds_test = ds.select(list(range(start_test_idx, len(ds))))

    all_ds[cls] = new_ds
    all_ds[cls + "_test"] = new_ds_test

data_dict = DatasetDict(all_ds)
data_dict.push_to_hub("stair-lab/dsa_hk231_wtc_per_student_sft_lf_splited")
