# 1. Load MIMIC-3 dataset and code descriptions
import pandas as pd
import json

# Load structured data from JSON
with open('', 'r') as file:
    data = json.load(file)

# SQUEL Commands
'''
-- Diagnoses
SELECT subject_id, hadm_id, icd9_code
FROM DIAGNOSES_ICD;

-- Procedures
SELECT subject_id, hadm_id, icd9_code
FROM PROCEDURES_ICD;

-- Medications
SELECT subject_id, hadm_id, ndc
FROM PRESCRIPTIONS;
'''


# Load code descriptions from MIMIC-3 CSV files
diagnosis_codes = pd.read_csv('path_to_D_ICD_DIAGNOSES.csv')
procedure_codes = pd.read_csv('path_to_D_ICD_PROCEDURES.csv')
medication_codes = pd.read_csv('path_to_PRESCRIPTIONS.csv')


# Create dictionaries for code to description mapping
diagnosis_dict = pd.Series(diagnosis_codes['LONG_TITLE'].values, index=diagnosis_codes['ICD9_CODE']).to_dict()
procedure_dict = pd.Series(procedure_codes['LONG_TITLE'].values, index=procedure_codes['ICD9_CODE']).to_dict()
medication_dict = pd.Series(medication_codes['DRUG_NAME_GENERIC'].values, index=medication_codes['NDC']).to_dict()

#------------------------------------------------------------------

# 2. Mapping from codes to their textual/semantic meaning 
''' Prompt Outline Example:
[INST]PATIENT HISTORY:
VISIT 1: 
  Date: 2023-01-10
  Diagnoses: Hypertension, Type 2 Diabetes
  Procedures: Blood Pressure Measurement, HbA1c Test
  Medications: Metformin, Lisinopril
Visit 2: 
  Diagnoses: Hypertension, Type 2 Diabetes, Hyperlipidemia
  Procedures: Lipid Panel, Blood Pressure Measurement
  Medications: Metformin, Lisinopril, Atorvastatin
  
NURSE NOTES:[/INST]
'''

# Function to convert codes to text
def convert_codes_to_text(codes, diagnosis_dict, procedure_dict, medication_dict):
    diagnoses = []
    procedures = []
    medications = []
    for code in codes:
        if code in diagnosis_dict:
            diagnoses.append(diagnosis_dict[code])
        elif code in procedure_dict:
            procedures.append(procedure_dict[code])
        elif code in medication_dict:
            medications.append(medication_dict[code])
    return diagnoses, procedures, medications

def generate_patient_narrative(record, diagnosis_dict, procedure_dict, medication_dict):
    narrative = ["PATIENT HISTORY:"]
    for idx, visit in enumerate(record['visits'], 1):
        diagnoses, procedures, medications = convert_codes_to_text(visit, diagnosis_dict, procedure_dict, medication_dict)
        visit_text = (
            f"Visit {idx}:\n"
            f"  Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}\n"
            f"  Procedures: {', '.join(procedures) if procedures else 'None'}\n"
            f"  Medications: {', '.join(medications) if medications else 'None'}"
        )
        narrative.append(visit_text)
    narrative.append("NURSE NOTES:")
    return "[INST]" + '\n'.join(narrative) + "[/INST]"




#------------------------------------------------------------------

# 3. Convert each record's structured data to narrative text format
narratives = [generate_patient_narrative(record, diagnosis_dict, procedure_dict, medication_dict) for record in data]

# Save narratives to new JSON file
with open('narratives.json', 'w') as file:
    json.dump(narratives, file)


