# MIMIC-IV Dataset Structure

## Overview

The MIMIC-IV database (Medical Information Mart for Intensive Care IV) is a large, freely-available database comprising de-identified health-related data associated with patients who stayed in critical care units at the Beth Israel Deaconess Medical Center. This document provides an overview of the structure and contents of MIMIC-IV version 3.1.

## Dataset Organization

MIMIC-IV is organized into two main modules:

1. **Hospital (`hosp`)** - Contains data related to patient hospital stays, including laboratory measurements, microbiology data, medication administration, and more.
2. **ICU (`icu`)** - Contains detailed data collected during ICU stays, including vital sign measurements, care procedures, and other ICU-specific information.

## Hospital (`hosp`) Module

The hospital module contains the following tables:

| Table | Description |
|-------|-------------|
| `admissions.csv` | Information about patient hospital admissions |
| `d_hcpcs.csv` | Dictionary of HCPCS (Healthcare Common Procedure Coding System) codes |
| `d_icd_diagnoses.csv` | Dictionary of ICD diagnosis codes |
| `d_icd_procedures.csv` | Dictionary of ICD procedure codes |
| `d_labitems.csv` | Dictionary of laboratory items |
| `diagnoses_icd.csv` | Hospital diagnoses coded using the ICD system |
| `drgcodes.csv` | Diagnosis Related Groups (DRG) codes |
| `emar.csv` | Electronic Medication Administration Records |
| `emar_detail.csv` | Detailed medication information related to electronic MAR |
| `hcpcsevents.csv` | HCPCS procedures |
| `labevents.csv` | Laboratory test results |
| `microbiologyevents.csv` | Microbiology test results |
| `pharmacy.csv` | Pharmacy medications and orders |
| `poe.csv` | Provider order entry data |
| `poe_detail.csv` | Detailed provider order information |
| `prescriptions.csv` | Medication prescriptions |
| `procedures_icd.csv` | Hospital procedures coded using the ICD system |
| `services.csv` | Hospital services (e.g., medicine, surgery, etc.) |
| `transfers.csv` | Patient transfers between hospital wards |

## ICU (`icu`) Module

The ICU module contains the following tables:

| Table | Description |
|-------|-------------|
| `caregiver.csv` | Information about caregivers |
| `chartevents.csv` | Charted observations and measurements for patients |
| `d_items.csv` | Dictionary of ICU items (e.g., vital signs, lab tests, etc.) |
| `datetimeevents.csv` | Events recorded with a date and time |
| `icustays.csv` | Information about patient ICU stays |
| `ingredientevents.csv` | Ingredients used in medications |
| `inputevents.csv` | Patient intake data (fluids, medications, etc.) |
| `outputevents.csv` | Patient output data (urine, drains, etc.) |
| `procedureevents.csv` | Procedures performed during ICU stay |

## Core Identifiers

Several identifiers are used to link tables across the database:

- `subject_id`: Unique identifier for a patient
- `hadm_id`: Unique identifier for a hospital admission
- `stay_id`: Unique identifier for an ICU stay
- `transfer_id`: Unique identifier for a patient transfer

## Table Relationships

- Patient data in all tables can be linked using `subject_id`
- Hospital admission data can be linked using `hadm_id`
- ICU stay data can be linked using `stay_id`

## File Format

Each table is available in both:
- `.csv` format (raw CSV)
- `.csv.gz` format (compressed CSV)

## Usage Notes

1. MIMIC-IV contains de-identified data in accordance with HIPAA Safe Harbor provisions.
2. Dates have been shifted to protect patient privacy, but time intervals remain intact.
3. All ages over 89 are aggregated into a single group labeled '90+'.
4. The dataset requires appropriate data use agreements and ethical considerations.

## Example Queries

### Getting admission information for a patient:
```sql
SELECT *
FROM hosp.admissions
WHERE subject_id = 10000032
ORDER BY admittime;
```

### Getting ICU stays for a specific hospital admission:
```sql
SELECT *
FROM icu.icustays
WHERE hadm_id = 29079034;
```

## Additional Resources

- [MIMIC-IV Documentation](https://mimic.mit.edu/docs/iv/)
- [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-IV Code Repository](https://github.com/MIT-LCP/mimic-code)
