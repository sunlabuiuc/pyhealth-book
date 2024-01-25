from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models import Transformer
from pyhealth.tasks import length_of_stay_prediction_mimic3_fn
import pickle

# STEP 1: load data
base_dataset = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    # code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
    dev=False,
    refresh_cache=False,
)
base_dataset.stat()

# STEP 2: set task
sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic3_fn)
sample_dataset.stat()

LENGTH = len(sample_dataset.samples)

train = []
val = []
test = []

for idx, sample in enumerate(sample_dataset.samples):
    if idx < 0.8 * LENGTH:
        train.append({
            "diagnoses": sample["conditions"],
            "length_of_stay": sample["label"],
        })
    elif idx < 0.9 * LENGTH:
        val.append({
            "diagnoses": sample["conditions"],
            "length_of_stay": sample["label"],
        })
    else:
        test.append({
            "diagnoses": sample["conditions"],
            "length_of_stay": sample["label"],
        })

print (len(train), len(val), len(test))

pickle.dump(train, open("train.pkl", "wb"))
pickle.dump(val, open("val.pkl", "wb"))
pickle.dump(test, open("test.pkl", "wb"))