import enum
from pathlib import Path
from typing import Literal


# Constants
DEFAULT_MIMIC_PATH      = Path("/Users/artinmajdi/Documents/GitHubs/RAP/mimic__pankaj/dataset/mimic-iv-3.1")
DEFAULT_NUM_SUBJECTS    = 10
RANDOM_STATE            = 42
SUBJECT_ID_COL          = 'subject_id'


class TableNamesHOSP(enum.Enum):
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

	@classmethod
	def values(cls):
		return [member.value for member in cls]

	@property
	def description(self):

		tables_descriptions = {
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
			('hosp', 'omr')               : "Order monitoring results"
		}

		return tables_descriptions.get(('hosp', self.value))

	@property
	def module(self):
		return 'hosp'

class TableNamesICU(enum.Enum):
	CAREGIVER          = 'caregiver'
	CHARTEVENTS        = 'chartevents'
	DATETIMEEVENTS     = 'datetimeevents'
	D_ITEMS            = 'd_items'
	ICUSTAYS           = 'icustays'
	INGREDIENTEVENTS   = 'ingredientevents'
	INPUTEVENTS        = 'inputevents'
	OUTPUTEVENTS       = 'outputevents'
	PROCEDUREEVENTS    = 'procedureevents'

	@classmethod
	def values(cls):
		return [member.value for member in cls]

	@property
	def description(self):

		tables_descriptions = {
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

		return tables_descriptions.get(('icu', self.value))

	@property
	def module(self):
		return 'icu'


def convert_table_names_to_enum_class(name: str, module: Literal['hosp', 'icu']='hosp') -> TableNamesHOSP | TableNamesICU:
	if module == 'hosp':
		return TableNamesHOSP(name)
	else:
		return TableNamesICU(name)


# Constants
dtypes_all = {
	'discontinued_by_poe_id': 'object',
	'long_description'      : 'string',
	'icd_code'              : 'string',
	'drg_type'              : 'category',
	'enter_provider_id'     : 'string',
	'hadm_id'               : 'int',
	'icustay_id'            : 'int',
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
	'infusion_type'         : 'object',
	'sliding_scale'         : 'object',
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
	'race'                  : 'category'}

parse_dates_all = [
			'admittime',
			'dischtime',
			'deathtime',
			'edregtime',
			'edouttime',
			'charttime',
			'scheduletime',
			'storetime',
			'storedate']


