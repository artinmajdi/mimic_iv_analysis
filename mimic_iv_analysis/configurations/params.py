import enum
from pathlib import Path
from typing import Literal
import pyarrow as pa
import pandas as pd
import dask.dataframe as dd

class TableNames(enum.Enum):

    # ----- Merged Tables -----
    MERGED = 'merged_table'

    # ----- HOSP Tables -----
    ADMISSIONS         = 'admissions'
    D_HCPCS            = 'd_hcpcs'
    D_ICD_DIAGNOSES    = 'd_icd_diagnoses'
    D_ICD_PROCEDURES   = 'd_icd_procedures'
    D_LABITEMS         = 'd_labitems'
    DIAGNOSES_ICD      = 'diagnoses_icd'
    DRGCODES           = 'drgcodes'
    EMAR               = 'emar'
    EMAR_DETAIL        = 'emar_detail'
    HCPCSEVENTS        = 'hcpcsevents'
    LABEVENTS          = 'labevents'
    MICROBIOLOGYEVENTS = 'microbiologyevents'
    OMR                = 'omr'
    PATIENTS           = 'patients'
    PHARMACY           = 'pharmacy'
    POE                = 'poe'
    POE_DETAIL         = 'poe_detail'
    PRESCRIPTIONS      = 'prescriptions'
    PROCEDURES_ICD     = 'procedures_icd'
    PROVIDER           = 'provider'
    SERVICES           = 'services'
    TRANSFERS          = 'transfers'

    # ----- ICU Tables -----
    CAREGIVER          = 'caregiver'
    CHARTEVENTS        = 'chartevents'
    DATETIMEEVENTS     = 'datetimeevents'
    D_ITEMS            = 'd_items'
    ICUSTAYS           = 'icustays'
    INGREDIENTEVENTS   = 'ingredientevents'
    INPUTEVENTS        = 'inputevents'
    OUTPUTEVENTS       = 'outputevents'
    PROCEDUREEVENTS    = 'procedureevents'

    @property
    def hosp(self):
        return ['ADMISSIONS', 'PATIENTS', 'LABEVENTS', 'MICROBIOLOGYEVENTS', 'PHARMACY', 'PRESCRIPTIONS', 'PROCEDURES_ICD', 'DIAGNOSES_ICD', 'EMAR', 'EMAR_DETAIL', 'POE', 'POE_DETAIL', 'D_HCPCS', 'D_ICD_DIAGNOSES', 'D_ICD_PROCEDURES', 'D_LABITEMS', 'HCPCSEVENTS', 'DRGCODES', 'SERVICES', 'TRANSFERS', 'PROVIDER']

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    @property
    def description(self):
        tables_descriptions = {

            # ----- Merged Tables -----
            ('merged', 'merged_table')    : "Merged table combining relevant columns from multiple HOSP and ICU sources.",

            # ----- HOSP Tables -----
            ('hosp', 'admissions')        : "Patient hospital admissions information",
            ('hosp', 'patients')          : "Patient demographic data",
            ('hosp', 'labevents')         : "Laboratory measurements (large file)",
            ('hosp', 'microbiologyevents'): "Microbiology test results",
            ('hosp', 'pharmacy')          : "Pharmacy orders",
            ('hosp', 'prescriptions')     : "Medication prescriptions",
            ('hosp', 'procedures_icd')    : "Patient procedures",
            ('hosp', 'diagnoses_icd')     : "Patient diagnoses",
            ('hosp', 'emar')              : "Electronic medication administration records",
            ('hosp', 'emar_detail')       : "Detailed medication administration data",
            ('hosp', 'poe')               : "Provider order entries",
            ('hosp', 'poe_detail')        : "Detailed order information",
            ('hosp', 'd_hcpcs')           : "HCPCS code definitions",
            ('hosp', 'd_icd_diagnoses')   : "ICD diagnosis code definitions",
            ('hosp', 'd_icd_procedures')  : "ICD procedure code definitions",
            ('hosp', 'd_labitems')        : "Laboratory test definitions",
            ('hosp', 'hcpcsevents')       : "HCPCS events",
            ('hosp', 'drgcodes')          : "Diagnosis-related group codes",
            ('hosp', 'services')          : "Hospital services",
            ('hosp', 'transfers')         : "Patient transfers",
            ('hosp', 'provider')          : "Provider information",
            ('hosp', 'omr')               : "Order monitoring results",

            # ----- ICU Tables -----
            ('icu', 'chartevents')        : "Patient charting data (vital signs, etc.)",
            ('icu', 'datetimeevents')     : "Date/time-based events",
            ('icu', 'inputevents')        : "Patient intake data",
            ('icu', 'outputevents')       : "Patient output data",
            ('icu', 'procedureevents')    : "ICU procedures",
            ('icu', 'ingredientevents')   : "Detailed medication ingredients",
            ('icu', 'd_items')            : "Dictionary of ICU items",
            ('icu', 'icustays')           : "ICU stay information",
            ('icu', 'caregiver')          : "Caregiver information"
        }
        return tables_descriptions.get((self.module, self.value))

    @property
    def module(self):
        if self == TableNames.MERGED:
            return 'merged'
        if self.name in self.hosp:
            return 'hosp'
        return 'icu'

DEFAULT_STUDY_TABLES_LIST = [
				TableNames.PATIENTS,
				TableNames.ADMISSIONS,
				TableNames.DIAGNOSES_ICD,
				TableNames.TRANSFERS,
				TableNames.D_ICD_DIAGNOSES,
				TableNames.POE,
				TableNames.POE_DETAIL,
                TableNames.PRESCRIPTIONS,
                TableNames.D_LABITEMS,
                TableNames.LABEVENTS,
			]

TABLES_W_SUBJECT_ID_COLUMN =  [
                    TableNames.MERGED,
                    TableNames.PATIENTS,
					TableNames.ADMISSIONS,
					TableNames.TRANSFERS,
					TableNames.DIAGNOSES_ICD,
					TableNames.POE,
					TableNames.POE_DETAIL,
					TableNames.MICROBIOLOGYEVENTS,
                    TableNames.LABEVENTS,
                    TableNames.PRESCRIPTIONS
                ]

# def convert_table_names_to_enum_class(name: str, module: Literal['hosp', 'icu']='hosp') -> TableNames:
#     return TableNames(name)

# TODO: add TableNames.PRESCRIPTIONS.value table
# Constants
# TODO: Remove all dependencies on DEFAULT_MIMIC_PATH path except for UI path selection default value.
# DEFAULT_MIMIC_PATH   = Path("/Users/artinmajdi/Documents/GitHubs/RAP/mimic__pankaj/dataset/mimic-iv-3.1")
DEFAULT_MIMIC_PATH   = Path("/Users/artinmajdi/Library/CloudStorage/GoogleDrive-msm2024@gmail.com/My Drive/MIMIC_IV_Dataset")
# DEFAULT_MIMIC_PATH   = Path("/Users/pankajvyas/MIMIC_Project/MIMIC-IV-Raw Data")
DEFAULT_NUM_SUBJECTS = 20
RANDOM_STATE         = 42
SUBJECT_ID_COL       = 'subject_id'


# Tables Details for Filtering
ADMISSION_LOCATIONS = ['TRANSFER FROM HOSPITAL', 'EMERGENCY ROOM', 'WALK-IN/SELF REFERRAL', 'PHYSICIAN REFERRAL']
ADMISSION_TYPES     = ['EMERGENCY', 'URGENT', 'ELECTIVE', 'NEWBORN', 'OBSERVATION']
ADMISSION_COLUMNS   = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admit_provider_id", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", "edregtime", "edouttime", "hospital_expire_flag"]

POE_COLUMNS = ['poe_id', 'poe_seq', 'subject_id', 'hadm_id', 'ordertime', 'order_type', 'order_subtype', 'transaction_type', 'discontinue_of_poe_id', 'discontinued_by_poe_id', 'order_provider_id', 'order_status']
POE_ORDER_TYPES = ['Medications', 'General Care', 'Nutrition', 'Blood Bank', 'Lab', 'Respiratory', 'ADT orders', 'Radiology', 'IV therapy', 'Consults']
POE_TRANSACTION_TYPES = ['NEW', 'D/C', 'Change']



# Type aliases
DataFrameType = pd.DataFrame | dd.DataFrame

# Updated dictionary for better Parquet compatibility
COLUMN_TYPES = {

    # ID columns
    'subject_id'            : 'int64',
    'stay_id'               : 'int64',
    'icustay_id'            : 'int64',
    'itemid'                : 'int64',
    'labevent_id'           : 'int64',
    'specimen_id'           : 'int64',
    'poe_seq'               : 'int64',
    'anchor_year'           : 'int64',
    'anchor_age'            : 'int64',
    'doses_per_24_hrs'      : 'string',

    'ab_name'               : 'string',
    'dilution_comparison'   : 'string',
    'dilution_text'         : 'string',


    'pharmacy_id'           : 'string',
    'poe_id'                : 'string',
    'order_provider_id'     : 'string',
    'enter_provider_id'     : 'string',
    'leave_provider_id'     : 'string',
    'dod'                   : 'string',
    'discontinued_by_poe_id': 'string',
    'discontinue_of_poe_id' : 'string',
    'drug'                  : 'string',
    'formulary_drug_cd'     : 'string',
    'gsn'                   : 'string',
    'ndc'                   : 'string',
    'prod_strength'         : 'string',
    'dose_val_rx'           : 'string',
    'dose_unit_rx'          : 'string',
    'form_rx'               : 'string',
    'form_val_disp'         : 'string',


    # Categorical columns
    'gender'              : 'category',
    'race'                : 'category',
    'marital_status'      : 'category',
    'admission_type'      : 'category',
    'admission_location'  : 'category',
    'insurance'           : 'category',
    'language'            : 'category',
    'discharge_location'  : 'category',
    'curr_service'        : 'category',
    'drug_type'           : 'category',
    'route'               : 'category',
    'form_rx'             : 'category',
    'dose_unit_rx'        : 'category',
    'form_unit_disp'      : 'category',
    'expiration_unit'     : 'category',
    'duration_interval'   : 'category',
    'dispensation'        : 'category',
    'drg_type'            : 'category',
    'hospital_expire_flag': 'category',
    'anchor_year_group'   : 'category',
    'icd_version'         : 'category',
    'seq_num'             : 'category',
    'form_unit_disp'      : 'category',
    'route'               : 'category',

    # Text/String columns
    'icd_code'   : 'string',
    'long_title' : 'string',
    'org_name'   : 'string',

    # Boolean columns
    'flag_mobil'     : 'bool',
    'flag_work_phone': 'bool',
    'flag_phone'     : 'bool',
    'flag_email'     : 'bool',

    # Object columns (requiring special handling)
    'interpretation'  : 'string',
    'quantity'        : 'string',
    'infusion_type'   : 'string',
    'sliding_scale'   : 'string',
    'fill_quantity'   : 'string',
    'expirationdate'  : 'string',
    'one_hr_max'      : 'string',
    'lockout_interval': 'string',
    'basal_rate'      : 'string',
    'form_val_disp'   : 'string',
    'gsn'             : 'string',
    'dose_val_rx'     : 'string',
    'prev_service'    : 'string',
    'hadm_id'         : 'string',
}

# List of datetime columns for parsing
DATETIME_COLUMNS = [
    'ordertime',
    'admittime',
    'dischtime',
    'deathtime',
    'edregtime',
    'edouttime',
    'charttime',
    'scheduletime',
    'storetime',
    'storedate',
    'starttime',
    'endtime',
    'transfertime',
    'stoptime'
]

# Mapping of tables to their categorical columns
TABLE_CATEGORICAL_COLUMNS = {
	TableNames.PATIENTS.value       : ['gender', 'race'],
	TableNames.ADMISSIONS.value     : ['admission_type', 'admission_location', 'discharge_location', 'insurance', 'language', 'marital_status'],
	TableNames.SERVICES.value       : ['curr_service', 'prev_service'],
	TableNames.PHARMACY.value       : ['drug_type', 'route', 'form_rx'],
	TableNames.PRESCRIPTIONS.value  : ['drug_type', 'route', 'form_rx', 'dose_unit_rx', 'form_unit_disp', 'expiration_unit', 'duration_interval', 'dispensation'],
	TableNames.DRGCODES.value       : ['drg_type'],
	TableNames.D_ICD_DIAGNOSES.value: ['icd_version'],
	TableNames.DIAGNOSES_ICD.value  : ['icd_version', 'seq_num'],
	TableNames.POE.value            : ['order_type', 'transaction_type']
}

# Default dtypes for pandas loading
dtypes_all = {
    'long_description'      : 'string',
    'icd_code'              : 'string',
    'drg_type'              : 'category',
    'enter_provider_id'     : 'string',
    'hadm_id'               : 'string', # TODO: make sure changing this from int to string didn't cause issues
    'icustay_id'            : 'int64',
    'leave_provider_id'     : 'string',
    'poe_id'                : 'string',
    'emar_id'               : 'string',
    'subject_id'            : 'int64',
    'pharmacy_id'           : 'string',
    'interpretation'        : 'object',
    'org_name'              : 'object',
    'quantity'              : 'object',
    'infusion_type'         : 'object',
    'sliding_scale'         : 'object',
    'fill_quantity'         : 'object',
    'expiration_unit'       : 'category',
    'duration_interval'     : 'category',
    'dispensation'          : 'category',
    'expirationdate'        : 'object',
    'one_hr_max'            : 'object',
    'lockout_interval'      : 'object',
    'basal_rate'            : 'object',
    'form_unit_disp'        : 'category',
    'route'                 : 'category',
    'dose_unit_rx'          : 'category',
    'drug_type'             : 'category',
    'form_rx'               : 'object',
    'form_val_disp'         : 'object',
    'gsn'                   : 'object',
    'dose_val_rx'           : 'object',
    'prev_service'          : 'object',
    'curr_service'          : 'category',
    'admission_type'        : 'category',
    'discharge_location'    : 'category',
    'insurance'             : 'category',
    'language'              : 'category',
    'marital_status'        : 'category',
    'race'                  : 'category'
}

# Dates for parsing when loading CSVs
parse_dates_all = DATETIME_COLUMNS


pyarrow_dtypes_map = {
	'int64'         : pa.int64(),
	'float64'       : pa.float64(),
	'bool'          : pa.bool_(),
	'datetime64[ns]': pa.timestamp('ns'),
	'category'      : pa.dictionary(pa.int32(), pa.string()),
	'object'        : pa.string(),
	'string'        : pa.string()
}
