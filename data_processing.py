import pandas as pd
import copy
from datetime import datetime
from tqdm import tqdm
import json
import os

root_dir = os.environ["MIMIC_ROOT"]
if root_dir is None:
    print("MIMIC_ROOT ENVIRONMENT VARIABLE NOT SET")
    exit(1)
file_directory = f"{root_dir}/1.4"

print("LOADING NOTES", datetime.now())
notes_path = f"{file_directory}/NOTEEVENTS.csv"
notes_df = pd.read_csv(notes_path)
notes_df = notes_df.loc[(notes_df["DESCRIPTION"] == "Report") & (notes_df["CATEGORY"] == "Nursing/other")]
notes_df["CHARTTIME"] = pd.to_datetime(notes_df["CHARTTIME"])
notes_df

print("LOADING DIAGNOSES", datetime.now())
diagnoses_path = f"{file_directory}/DIAGNOSES_ICD.csv"
diagnoses_df = pd.read_csv(diagnoses_path)
diagnoses_df

print("LOADING ICD DICT", datetime.now())
icd_dict_path = f"{file_directory}/D_ICD_DIAGNOSES.csv"
icd_dict_df = pd.read_csv(icd_dict_path)
icd_dict_df = icd_dict_df.set_index("ICD9_CODE")
icd_dict_df

print("LOADING ADMISSIONS", datetime.now())
admissions_path = f"{file_directory}/ADMISSIONS.csv"
admissions_df = pd.read_csv(admissions_path)
admissions_df["ADMITTIME"] = pd.to_datetime(admissions_df["ADMITTIME"])
admissions_df

print("DONE LOADING CSVs", datetime.now())


subjects = set(admissions_df["SUBJECT_ID"])


datapoints = []
for idx, subject in tqdm(enumerate(subjects), total=len(subjects)):
    subj_admissions = admissions_df.loc[admissions_df["SUBJECT_ID"] == subject]
    subj_admissions = subj_admissions.sort_values(by="ADMITTIME")
    # print("Associated subj_admissinos")
    # display(subj_admissions)
    
    history_so_far = []
    for i, row in subj_admissions.iterrows():
        current_hadm_id = row["HADM_ID"]
        # print("Examinng visit", current_hadm_id)

        admission_diagnoses = diagnoses_df.loc[diagnoses_df["HADM_ID"] == current_hadm_id]
        admission_diagnoses_codes = admission_diagnoses["ICD9_CODE"]
        history_so_far.append(list(admission_diagnoses_codes))
        # print("History so far", history_so_far)

        notes_from_this_visit = notes_df.loc[notes_df["HADM_ID"] == current_hadm_id]
        # print("Notes from visit")
        # display(notes_from_this_visit)
        if len(notes_from_this_visit.index) > 0:
            notes_from_this_visit = notes_from_this_visit.sort_values(by="CHARTTIME")

            combined_notes = "\n".join(notes_from_this_visit["TEXT"])
            data_point = (copy.deepcopy(history_so_far), combined_notes)
            datapoints.append(data_point)


renamed_datapoints = []
for datapoint in datapoints:
    history_so_far = datapoint[0]
    combined_notes = datapoint[1]

    renamed_history_so_far = []
    for visit in history_so_far:
        renamed_visit = []
        for code in visit:
            if code in icd_dict_df.index:
                code_name = icd_dict_df.loc[code]["LONG_TITLE"]
                renamed_visit.append(code_name)
        renamed_history_so_far.append(renamed_visit)

    renamed_datapoints.append((renamed_history_so_far, combined_notes))


output_file = f"{root_dir}/extracted_dataset.json"
with open(output_file, "w") as f:
    json.dump(renamed_datapoints, f)