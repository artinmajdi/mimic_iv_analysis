import enum
from pathlib import Path
import pyarrow as pa
import pandas as pd
import dask.dataframe as dd


class ColumnNames(enum.Enum):
    """
    Enumeration of all MIMIC-IV column names.

    This enum provides a centralized way to reference column names throughout the codebase,
    ensuring consistency and reducing the risk of typos. All column names from the MIMIC-IV
    database are included as enum members.
    """

    # Column names (alphabetically ordered)
    AB_ITEMID                            = 'ab_itemid'
    AB_NAME                              = 'ab_name'
    ADMINISTRATION_TYPE                  = 'administration_type'
    ADMISSION_LOCATION                   = 'admission_location'
    ADMISSION_TYPE                       = 'admission_type'
    ADMIT_PROVIDER_ID                    = 'admit_provider_id'
    ADMITTIME                            = 'admittime'
    ANCHOR_AGE                           = 'anchor_age'
    ANCHOR_YEAR                          = 'anchor_year'
    ANCHOR_YEAR_GROUP                    = 'anchor_year_group'
    BARCODE_TYPE                         = 'barcode_type'
    BASAL_RATE                           = 'basal_rate'
    CAREUNIT                             = 'careunit'
    CHARTDATE                            = 'chartdate'
    CHARTTIME                            = 'charttime'
    COMMENTS                             = 'comments'
    COMPLETE_DOSE_NOT_GIVEN              = 'complete_dose_not_given'
    COMPLETION_INTERVAL                  = 'completion_interval'
    CONTINUED_INFUSION_IN_OTHER_LOCATION = 'continued_infusion_in_other_location'
    CURR_SERVICE                         = 'curr_service'
    DEATHTIME                            = 'deathtime'
    DESCRIPTION                          = 'description'
    DILUTION_COMPARISON                  = 'dilution_comparison'
    DILUTION_TEXT                        = 'dilution_text'
    DILUTION_VALUE                       = 'dilution_value'
    DISCONTINUE_OF_POE_ID                = 'discontinue_of_poe_id'
    DISCONTINUED_BY_POE_ID               = 'discontinued_by_poe_id'
    DISCHARGE_LOCATION                   = 'discharge_location'
    DISCHTIME                            = 'dischtime'
    DISP_SCHED                           = 'disp_sched'
    DISPENSATION                         = 'dispensation'
    DOD                                  = 'dod'
    DOSE_DUE                             = 'dose_due'
    DOSE_DUE_UNIT                        = 'dose_due_unit'
    DOSE_GIVEN                           = 'dose_given'
    DOSE_GIVEN_UNIT                      = 'dose_given_unit'
    DOSE_UNIT_RX                         = 'dose_unit_rx'
    DOSE_VAL_RX                          = 'dose_val_rx'
    DOSES_PER_24_HRS                     = 'doses_per_24_hrs'
    DRG_CODE                             = 'drg_code'
    DRG_MORTALITY                        = 'drg_mortality'
    DRG_SEVERITY                         = 'drg_severity'
    DRG_TYPE                             = 'drg_type'
    DRUG                                 = 'drug'
    DRUG_TYPE                            = 'drug_type'
    DURATION                             = 'duration'
    DURATION_INTERVAL                    = 'duration_interval'
    EDOUTTIME                            = 'edouttime'
    EDREGTIME                            = 'edregtime'
    EMAR_ID                              = 'emar_id'
    EMAR_SEQ                             = 'emar_seq'
    ENTER_PROVIDER_ID                    = 'enter_provider_id'
    ENTERTIME                            = 'entertime'
    EVENT_TXT                            = 'event_txt'
    EVENTTYPE                            = 'eventtype'
    EXPIRATION_UNIT                      = 'expiration_unit'
    EXPIRATION_VALUE                     = 'expiration_value'
    EXPIRATIONDATE                       = 'expirationdate'
    FIELD_NAME                           = 'field_name'
    FIELD_VALUE                          = 'field_value'
    FILL_QUANTITY                        = 'fill_quantity'
    FLAG                                 = 'flag'
    FLAG_EMAIL                           = 'flag_email'
    FLAG_MOBIL                           = 'flag_mobil'
    FLAG_PHONE                           = 'flag_phone'
    FLAG_WORK_PHONE                      = 'flag_work_phone'
    FORM_RX                              = 'form_rx'
    FORM_UNIT_DISP                       = 'form_unit_disp'
    FORM_VAL_DISP                        = 'form_val_disp'
    FORMULARY_DRUG_CD                    = 'formulary_drug_cd'
    FREQUENCY                            = 'frequency'
    GENDER                               = 'gender'
    GSN                                  = 'gsn'
    HADM_ID                              = 'hadm_id'
    HCPCS_CD                             = 'hcpcs_cd'
    HOSPITAL_EXPIRE_FLAG                 = 'hospital_expire_flag'
    ICD_CODE                             = 'icd_code'
    ICD_VERSION                          = 'icd_version'
    ICUSTAY_ID                           = 'icustay_id'
    INFUSION_COMPLETE                    = 'infusion_complete'
    INFUSION_RATE                        = 'infusion_rate'
    INFUSION_RATE_ADJUSTMENT             = 'infusion_rate_adjustment'
    INFUSION_RATE_ADJUSTMENT_AMOUNT      = 'infusion_rate_adjustment_amount'
    INFUSION_RATE_UNIT                   = 'infusion_rate_unit'
    INFUSION_TYPE                        = 'infusion_type'
    INSURANCE                            = 'insurance'
    INTERPRETATION                       = 'interpretation'
    INTIME                               = 'intime'
    ISOLATE_NUM                          = 'isolate_num'
    ITEMID                               = 'itemid'
    LABEVENT_ID                          = 'labevent_id'
    LANGUAGE                             = 'language'
    LEAVE_PROVIDER_ID                    = 'leave_provider_id'
    LOCKOUT_INTERVAL                     = 'lockout_interval'
    LONG_TITLE                           = 'long_title'
    MARITAL_STATUS                       = 'marital_status'
    MEDICATION                           = 'medication'
    MICRO_SPECIMEN_ID                    = 'micro_specimen_id'
    MICROEVENT_ID                        = 'microevent_id'
    NDC                                  = 'ndc'
    NEW_IV_BAG_HUNG                      = 'new_iv_bag_hung'
    NON_FORMULARY_VISUAL_VERIFICATION    = 'non_formulary_visual_verification'
    ONE_HR_MAX                           = 'one_hr_max'
    ORDER_PROVIDER_ID                    = 'order_provider_id'
    ORDER_STATUS                         = 'order_status'
    ORDER_SUBTYPE                        = 'order_subtype'
    ORDER_TYPE                           = 'order_type'
    ORDERTIME                            = 'ordertime'
    ORG_ITEMID                           = 'org_itemid'
    ORG_NAME                             = 'org_name'
    OUTTIME                              = 'outtime'
    PARENT_FIELD_ORDINAL                 = 'parent_field_ordinal'
    PHARMACY_ID                          = 'pharmacy_id'
    POE_ID                               = 'poe_id'
    POE_SEQ                              = 'poe_seq'
    PREV_SERVICE                         = 'prev_service'
    PRIOR_INFUSION_RATE                  = 'prior_infusion_rate'
    PRIORITY                             = 'priority'
    PROC_TYPE                            = 'proc_type'
    PROD_STRENGTH                        = 'prod_strength'
    PRODUCT_AMOUNT_GIVEN                 = 'product_amount_given'
    PRODUCT_CODE                         = 'product_code'
    PRODUCT_DESCRIPTION                  = 'product_description'
    PRODUCT_DESCRIPTION_OTHER            = 'product_description_other'
    PRODUCT_UNIT                         = 'product_unit'
    PROVIDER_ID                          = 'provider_id'
    QUANTITY                             = 'quantity'
    RACE                                 = 'race'
    REASON_FOR_NO_BARCODE                = 'reason_for_no_barcode'
    REF_RANGE_LOWER                      = 'ref_range_lower'
    REF_RANGE_UPPER                      = 'ref_range_upper'
    RESTART_INTERVAL                     = 'restart_interval'
    ROUTE                                = 'route'
    SCHEDULETIME                         = 'scheduletime'
    SEQ_NUM                              = 'seq_num'
    SHORT_DESCRIPTION                    = 'short_description'
    SIDE                                 = 'side'
    SITE                                 = 'site'
    SLIDING_SCALE                        = 'sliding_scale'
    SPEC_ITEMID                          = 'spec_itemid'
    SPEC_TYPE_DESC                       = 'spec_type_desc'
    SPECIMEN_ID                          = 'specimen_id'
    STARTTIME                            = 'starttime'
    STATUS                               = 'status'
    STAY_ID                              = 'stay_id'
    STOPTIME                             = 'stoptime'
    STOREDATE                            = 'storedate'
    STORETIME                            = 'storetime'
    SUBJECT_ID                           = 'subject_id'
    TEST_NAME                            = 'test_name'
    TEST_SEQ                             = 'test_seq'
    TRANSACTION_TYPE                     = 'transaction_type'
    TRANSFER_ID                          = 'transfer_id'
    TRANSFERTIME                         = 'transfertime'
    VALUE                                = 'value'
    VALUENUM                             = 'valuenum'
    VALUEUOM                             = 'valueuom'
    VERIFIEDTIME                         = 'verifiedtime'
    WILL_REMAINDER_OF_DOSE_BE_GIVEN      = 'will_remainder_of_dose_be_given'


class TableNames(enum.Enum):
    """
    Enumeration of all MIMIC-IV table names with their corresponding file names.

    This enum provides a centralized way to reference table names throughout the codebase,
    ensuring consistency and reducing the risk of typos. Tables are organized by module
    (HOSP and ICU) and include utility methods for table discovery and column mapping.
    """

    # Special Tables
    MERGED = 'merged_table'

    # HOSP Module Tables
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

    # ICU Module Tables
    CAREGIVER        = 'caregiver'
    CHARTEVENTS      = 'chartevents'
    DATETIMEEVENTS   = 'datetimeevents'
    D_ITEMS          = 'd_items'
    ICUSTAYS         = 'icustays'
    INGREDIENTEVENTS = 'ingredientevents'
    INPUTEVENTS      = 'inputevents'
    OUTPUTEVENTS     = 'outputevents'
    PROCEDUREEVENTS  = 'procedureevents'


_HOSP_TABLES = frozenset([
    'admissions', 'd_hcpcs', 'd_icd_diagnoses', 'd_icd_procedures', 'd_labitems',
    'diagnoses_icd', 'drgcodes', 'emar', 'emar_detail', 'hcpcsevents', 'labevents',
    'microbiologyevents', 'omr', 'patients', 'pharmacy', 'poe', 'poe_detail',
    'prescriptions', 'procedures_icd', 'provider', 'services', 'transfers' ])

_ICU_TABLES = frozenset([
    'caregiver', 'chartevents', 'datetimeevents', 'd_items', 'icustays',
    'ingredientevents', 'inputevents', 'outputevents', 'procedureevents' ])

_COLUMN_TO_TABLES = {
    'ab_itemid'             : ['microbiologyevents'],
    'ab_name'               : ['microbiologyevents'],
    'admission_location'    : ['admissions'],
    'admission_type'        : ['admissions'],
    'admit_provider_id'     : ['admissions'],
    'admittime'             : ['admissions'],
    'anchor_age'            : ['patients'],
    'anchor_year'           : ['patients'],
    'anchor_year_group'     : ['patients'],
    'basal_rate'            : ['pharmacy'],
    'careunit'              : ['transfers'],
    'chartdate'             : ['microbiologyevents', 'hcpcsevents'],
    'charttime'             : ['labevents', 'microbiologyevents', 'emar'],
    'comments'              : ['labevents', 'microbiologyevents'],
    'curr_service'          : ['services'],
    'deathtime'             : ['admissions'],
    'description'           : ['drgcodes'],
    'dilution_comparison'   : ['microbiologyevents'],
    'dilution_text'         : ['microbiologyevents'],
    'dilution_value'        : ['microbiologyevents'],
    'discontinue_of_poe_id' : ['poe'],
    'discontinued_by_poe_id': ['poe'],
    'discharge_location'    : ['admissions'],
    'dischtime'             : ['admissions'],
    'disp_sched'            : ['pharmacy'],
    'dispensation'          : ['pharmacy'],
    'dod'                   : ['patients'],
    'dose_unit_rx'          : ['prescriptions'],
    'dose_val_rx'           : ['prescriptions'],
    'doses_per_24_hrs'      : ['prescriptions', 'pharmacy'],
    'drg_code'              : ['drgcodes'],
    'drg_mortality'         : ['drgcodes'],
    'drg_severity'          : ['drgcodes'],
    'drg_type'              : ['drgcodes'],
    'drug'                  : ['prescriptions'],
    'drug_type'             : ['prescriptions'],
    'duration'              : ['pharmacy'],
    'duration_interval'     : ['pharmacy'],
    'edouttime'             : ['admissions'],
    'edregtime'             : ['admissions'],
    'emar_id'               : ['emar', 'emar_detail'],
    'emar_seq'              : ['emar'],
    'enter_provider_id'     : ['emar'],
    'entertime'             : ['pharmacy'],
    'event_txt'             : ['emar'],
    'eventtype'             : ['transfers'],
    'expiration_unit'       : ['pharmacy'],
    'expiration_value'      : ['pharmacy'],
    'expirationdate'        : ['pharmacy'],
    'field_name'            : ['poe_detail', 'emar_detail'],
    'field_value'           : ['poe_detail', 'emar_detail'],
    'fill_quantity'         : ['pharmacy'],
    'flag'                  : ['labevents'],
    'form_rx'               : ['prescriptions'],
    'form_unit_disp'        : ['prescriptions'],
    'form_val_disp'         : ['prescriptions'],
    'frequency'             : ['pharmacy'],
    'gender'                : ['patients'],
    'gsn'                   : ['prescriptions'],
    'hadm_id'               : ['admissions', 'transfers', 'diagnoses_icd', 'procedures_icd', 'labevents', 'prescriptions', 'microbiologyevents', 'pharmacy', 'poe', 'hcpcsevents', 'drgcodes', 'emar', 'services'],
    'hcpcs_cd'              : ['hcpcsevents'],
    'hospital_expire_flag'  : ['admissions'],
    'icd_code'              : ['diagnoses_icd', 'procedures_icd'],
    'icd_version'           : ['diagnoses_icd', 'procedures_icd'],
    'infusion_type'         : ['pharmacy'],
    'insurance'             : ['admissions'],
    'interpretation'        : ['microbiologyevents'],
    'intime'                : ['transfers'],
    'isolate_num'           : ['microbiologyevents'],
    'itemid'                : ['labevents'],
    'labevent_id'           : ['labevents'],
    'language'              : ['admissions'],
    'lockout_interval'      : ['pharmacy'],
    'marital_status'        : ['admissions'],
    'medication'            : ['pharmacy', 'emar'],
    'microevent_id'         : ['microbiologyevents'],
    'ndc'                   : ['prescriptions'],
    'one_hr_max'            : ['pharmacy'],
    'order_provider_id'     : ['labevents', 'prescriptions', 'microbiologyevents', 'poe'],
    'order_status'          : ['poe'],
    'order_subtype'         : ['poe'],
    'order_type'            : ['poe'],
    'ordertime'             : ['poe'],
    'org_itemid'            : ['microbiologyevents'],
    'org_name'              : ['microbiologyevents'],
    'outtime'               : ['transfers'],
    'parent_field_ordinal'  : ['emar_detail'],
    'pharmacy_id'           : ['prescriptions', 'pharmacy', 'emar'],
    'poe_id'                : ['prescriptions', 'pharmacy', 'poe', 'poe_detail', 'emar'],
    'poe_seq'               : ['poe', 'poe_detail'],
    'prev_service'          : ['services'],
    'priority'              : ['labevents'],
    'proc_type'             : ['pharmacy'],
    'prod_strength'         : ['prescriptions'],
    'provider_id'           : ['provider'],
    'quantity'              : ['microbiologyevents'],
    'race'                  : ['admissions'],
    'ref_range_lower'       : ['labevents'],
    'ref_range_upper'       : ['labevents'],
    'route'                 : ['prescriptions', 'pharmacy'],
    'scheduletime'          : ['emar'],
    'seq_num'               : ['diagnoses_icd', 'procedures_icd', 'hcpcsevents'],
    'short_description'     : ['hcpcsevents'],
    'sliding_scale'         : ['pharmacy'],
    'spec_itemid'           : ['microbiologyevents'],
    'spec_type_desc'        : ['microbiologyevents'],
    'starttime'             : ['prescriptions', 'pharmacy'],
    'status'                : ['pharmacy'],
    'stoptime'              : ['prescriptions', 'pharmacy'],
    'storedate'             : ['microbiologyevents'],
    'storetime'             : ['labevents', 'microbiologyevents', 'emar'],
    'subject_id'            : ['admissions', 'patients', 'transfers', 'diagnoses_icd', 'procedures_icd', 'labevents', 'prescriptions', 'microbiologyevents', 'pharmacy', 'poe', 'poe_detail', 'hcpcsevents', 'drgcodes', 'emar', 'emar_detail', 'services'],
    'test_seq'              : ['microbiologyevents'],
    'transaction_type'      : ['poe'],
    'transfer_id'           : ['transfers'],
    'transfertime'          : ['services'],
    'value'                 : ['labevents'],
    'valuenum'              : ['labevents'],
    'valueuom'              : ['labevents'],
    'verifiedtime'          : ['pharmacy']
    }

_TABLE_TO_COLUMNS = {
    'admissions'        : ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'admit_provider_id', 'admission_location', 'discharge_location', 'insurance', 'language', 'marital_status', 'race', 'edregtime', 'edouttime', 'hospital_expire_flag'],
    'patients'          : ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod'],
    'transfers'         : ['subject_id', 'hadm_id', 'transfer_id', 'eventtype', 'careunit', 'intime', 'outtime'],
    'diagnoses_icd'     : ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'],
    'procedures_icd'    : ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'],
    'labevents'         : ['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments', 'order_provider_id'],
    'prescriptions'     : ['subject_id', 'hadm_id', 'pharmacy_id', 'poe_id', 'poe_seq', 'order_provider_id', 'starttime', 'stoptime', 'drug_type', 'drug', 'formulary_drug_cd', 'gsn', 'ndc', 'prod_strength', 'form_rx', 'dose_unit_rx', 'dose_val_rx', 'form_unit_disp', 'form_val_disp', 'doses_per_24_hrs', 'route'],
    'microbiologyevents': ['microevent_id', 'subject_id', 'hadm_id', 'micro_specimen_id', 'order_provider_id', 'chartdate', 'charttime', 'spec_itemid', 'spec_type_desc', 'test_seq', 'storedate', 'storetime', 'test_name', 'org_itemid', 'org_name', 'isolate_num', 'quantity', 'ab_itemid', 'ab_name', 'dilution_text', 'dilution_comparison', 'dilution_value', 'interpretation', 'comments'],
    'pharmacy'          : ['subject_id', 'hadm_id', 'pharmacy_id', 'poe_id', 'starttime', 'stoptime', 'medication', 'proc_type', 'status', 'entertime', 'verifiedtime', 'route', 'frequency', 'disp_sched', 'infusion_type', 'sliding_scale', 'lockout_interval', 'basal_rate', 'one_hr_max', 'doses_per_24_hrs', 'duration', 'duration_interval', 'expiration_value', 'expiration_unit', 'expirationdate', 'dispensation', 'fill_quantity'],
    'poe'               : ['poe_id', 'poe_seq', 'subject_id', 'hadm_id', 'ordertime', 'order_type', 'order_subtype', 'transaction_type', 'discontinue_of_poe_id', 'discontinued_by_poe_id', 'order_provider_id', 'order_status'],
    'poe_detail'        : ['poe_id', 'poe_seq', 'subject_id', 'field_name', 'field_value'],
    'hcpcsevents'       : ['subject_id', 'hadm_id', 'chartdate', 'hcpcs_cd', 'seq_num', 'short_description'],
    'drgcodes'          : ['subject_id', 'hadm_id', 'drg_type', 'drg_code', 'description', 'drg_severity', 'drg_mortality'],
    'emar'              : ['subject_id', 'hadm_id', 'emar_id', 'emar_seq', 'poe_id', 'pharmacy_id', 'enter_provider_id', 'charttime', 'medication', 'event_txt', 'scheduletime', 'storetime'],
    'emar_detail'       : ['subject_id', 'emar_id', 'emar_seq', 'parent_field_ordinal', 'administration_type', 'pharmacy_id', 'barcode_type', 'reason_for_no_barcode', 'complete_dose_not_given', 'dose_due', 'dose_due_unit', 'dose_given', 'dose_given_unit', 'will_remainder_of_dose_be_given', 'product_amount_given', 'product_unit', 'product_code', 'product_description', 'product_description_other', 'prior_infusion_rate', 'infusion_rate', 'infusion_rate_adjustment', 'infusion_rate_adjustment_amount', 'infusion_rate_unit', 'route', 'infusion_complete', 'completion_interval', 'new_iv_bag_hung', 'continued_infusion_in_other_location', 'restart_interval', 'side', 'site', 'non_formulary_visual_verification'],
    'services'          : ['subject_id', 'hadm_id', 'transfertime', 'prev_service', 'curr_service'],
    'provider'          : ['provider_id']
    }

_COLUMN_TYPES = {
    # Integer columns
    'subject_id'          : 'int64',   # INTEGER NOT NULL in MIMIC-IV
    'hadm_id'             : 'int64',   # INTEGER NOT NULL in MIMIC-IV (was incorrectly 'string')
    'stay_id'             : 'int64',
    'icustay_id'          : 'int64',
    'itemid'              : 'Int64',   # INTEGER - labevents
    'labevent_id'         : 'int64',   # INTEGER NOT NULL in MIMIC-IV
    'specimen_id'         : 'int64',   # INTEGER NOT NULL in MIMIC-IV
    'poe_seq'             : 'int64',   # INTEGER - poe, poe_detail
    'anchor_year'         : 'int64',   # INTEGER - patients
    'anchor_age'          : 'int64',   # INTEGER - patients
    'hospital_expire_flag': 'int16',   # INTEGER - admissions
    'ab_itemid'           : 'int64',   # INTEGER - microbiologyevents
    'emar_seq'            : 'int64',   # INTEGER - emar
    'isolate_num'         : 'int64',   # INTEGER - microbiologyevents
    'microevent_id'       : 'int64',   # INTEGER - microbiologyevents
    'org_itemid'          : 'int64',   # INTEGER - microbiologyevents
    'seq_num'             : 'int64',   # INTEGER - diagnoses_icd, procedures_icd, hcpcsevents
    'spec_itemid'         : 'int64',   # INTEGER - microbiologyevents
    'test_seq'            : 'int64',   # INTEGER - microbiologyevents
    'transfer_id'         : 'int64',   # INTEGER - transfers
    'drg_mortality'       : 'int64',   # INTEGER - drgcodes
    'drg_severity'        : 'int64',   # INTEGER - drgcodes
    'expiration_value'    : 'int64',   # INTEGER - pharmacy
    'icd_version'         : 'int64',   # INTEGER - diagnoses_icd, procedures_icd
    'pharmacy_id'         : 'int64',   # INTEGER - prescriptions, pharmacy, emar

    # Float/Double columns
    'basal_rate'          : 'float64', # DOUBLE - pharmacy
    'dilution_value'      : 'float64', # DOUBLE - microbiologyevents
    'doses_per_24_hrs'    : 'string', # DOUBLE - prescriptions, pharmacy # sold actually be float64 but some rows are of string type (eg 2.1-4)
    'duration'            : 'float64', # DOUBLE - pharmacy
    'one_hr_max'          : 'float64', # DOUBLE - pharmacy
    'parent_field_ordinal': 'float64', # DOUBLE - emar_detail
    'ref_range_lower'     : 'float64', # DOUBLE - labevents
    'ref_range_upper'     : 'float64', # DOUBLE - labevents
    'valuenum'            : 'float64', # DOUBLE - labevents

    # String/VARCHAR columns
    'ab_name'               : 'string', # VARCHAR - microbiologyevents
    'admission_location'    : 'string', # VARCHAR - admissions
    'admission_type'        : 'string', # VARCHAR - admissions
    'admit_provider_id'     : 'string', # VARCHAR - admissions
    'anchor_year_group'     : 'string', # VARCHAR - patients
    'careunit'              : 'string', # VARCHAR - transfers
    'comments'              : 'string', # VARCHAR - labevents, microbiologyevents
    'curr_service'          : 'string', # VARCHAR - services
    'description'           : 'string', # VARCHAR - drgcodes
    'dilution_comparison'   : 'string', # VARCHAR - microbiologyevents
    'dilution_text'         : 'string', # VARCHAR - microbiologyevents
    'discontinue_of_poe_id' : 'string', # VARCHAR - poe
    'discontinued_by_poe_id': 'string', # VARCHAR - poe
    'discharge_location'    : 'string', # VARCHAR - admissions
    'disp_sched'            : 'string', # VARCHAR - pharmacy
    'dispensation'          : 'string', # VARCHAR - pharmacy
    'dose_unit_rx'          : 'string', # VARCHAR - prescriptions
    'dose_val_rx'           : 'string', # VARCHAR - prescriptions
    'drg_code'              : 'string', # VARCHAR - drgcodes
    'drg_type'              : 'string', # VARCHAR - drgcodes
    'drug'                  : 'string', # VARCHAR - prescriptions
    'drug_type'             : 'string', # VARCHAR - prescriptions
    'duration_interval'     : 'string', # VARCHAR - pharmacy
    'emar_id'               : 'string', # VARCHAR - emar, emar_detail
    'enter_provider_id'     : 'string', # VARCHAR - emar
    'event_txt'             : 'string', # VARCHAR - emar
    'eventtype'             : 'string', # VARCHAR - transfers
    'expiration_unit'       : 'string', # VARCHAR - pharmacy
    'field_name'            : 'string', # VARCHAR - poe_detail, emar_detail
    'field_value'           : 'string', # VARCHAR - poe_detail, emar_detail
    'fill_quantity'         : 'string', # VARCHAR - pharmacy
    'flag'                  : 'string', # VARCHAR - labevents
    'form_rx'               : 'string', # VARCHAR - prescriptions
    'form_unit_disp'        : 'string', # VARCHAR - prescriptions
    'form_val_disp'         : 'string', # VARCHAR - prescriptions
    'frequency'             : 'string', # VARCHAR - pharmacy
    'gender'                : 'string', # VARCHAR - patients
    'gsn'                   : 'string', # VARCHAR - prescriptions
    'hcpcs_cd'              : 'string', # VARCHAR - hcpcsevents
    'icd_code'              : 'string', # VARCHAR - diagnoses_icd, procedures_icd
    'infusion_type'         : 'string', # VARCHAR - pharmacy
    'insurance'             : 'string', # VARCHAR - admissions
    'interpretation'        : 'string', # VARCHAR - microbiologyevents
    'language'              : 'string', # VARCHAR - admissions
    'lockout_interval'      : 'string', # VARCHAR - pharmacy
    'marital_status'        : 'string', # VARCHAR - admissions
    'medication'            : 'string', # VARCHAR - pharmacy, emar
    'ndc'                   : 'string', # VARCHAR - prescriptions
    'order_provider_id'     : 'string', # VARCHAR - labevents, prescriptions, microbiologyevents, poe
    'order_status'          : 'string', # VARCHAR - poe
    'order_subtype'         : 'string', # VARCHAR - poe
    'order_type'            : 'string', # VARCHAR - poe
    'org_name'              : 'string', # VARCHAR - microbiologyevents
    'poe_id'                : 'string', # VARCHAR - prescriptions, pharmacy, poe, poe_detail, emar
    'prev_service'          : 'string', # VARCHAR - services
    'priority'              : 'string', # VARCHAR - labevents
    'proc_type'             : 'string', # VARCHAR - pharmacy
    'prod_strength'         : 'string', # VARCHAR - prescriptions
    'provider_id'           : 'string', # VARCHAR - provider
    'quantity'              : 'string', # VARCHAR - microbiologyevents
    'race'                  : 'string', # VARCHAR - admissions
    'route'                 : 'string', # VARCHAR - prescriptions, pharmacy
    'short_description'     : 'string', # VARCHAR - hcpcsevents
    'sliding_scale'         : 'string', # VARCHAR - pharmacy
    'spec_type_desc'        : 'string', # VARCHAR - microbiologyevents
    'status'                : 'string', # VARCHAR - pharmacy
    'transaction_type'      : 'string', # VARCHAR - poe
    'value'                 : 'string', # VARCHAR - labevents
    'valueuom'              : 'string', # VARCHAR - labevents

    # Legacy columns (keeping for backward compatibility)
    'formulary_drug_cd'     : 'string',
    'long_title'            : 'string',
    'leave_provider_id'     : 'string',

    # DateTime columns
    'admittime'       : 'datetime64[ns]', # DATETIME - admissions
    'chartdate'       : 'datetime64[ns]', # DATETIME - microbiologyevents, hcpcsevents
    'charttime'       : 'datetime64[ns]', # DATETIME - labevents, microbiologyevents, emar
    'deathtime'       : 'datetime64[ns]', # DATETIME - admissions
    'dischtime'       : 'datetime64[ns]', # DATETIME - admissions
    'dod'             : 'datetime64[ns]', # DATETIME - patients
    'edouttime'       : 'datetime64[ns]', # DATETIME - admissions
    'edregtime'       : 'datetime64[ns]', # DATETIME - admissions
    'entertime'       : 'datetime64[ns]', # DATETIME - pharmacy
    'expirationdate'  : 'datetime64[ns]', # DATETIME - pharmacy
    'intime'          : 'datetime64[ns]', # DATETIME - transfers
    'ordertime'       : 'datetime64[ns]', # DATETIME - poe
    'outtime'         : 'datetime64[ns]', # DATETIME - transfers
    'scheduletime'    : 'datetime64[ns]', # DATETIME - emar
    'starttime'       : 'datetime64[ns]', # DATETIME - prescriptions, pharmacy
    'stoptime'        : 'datetime64[ns]', # DATETIME - prescriptions, pharmacy
    'storedate'       : 'datetime64[ns]', # DATETIME - microbiologyevents
    'storetime'       : 'datetime64[ns]', # DATETIME - labevents, microbiologyevents, emar
    'transfertime'    : 'datetime64[ns]', # DATETIME - services
    'verifiedtime'    : 'datetime64[ns]', # DATETIME - pharmacy


    # Boolean columns
    'flag_mobil'     : 'bool',
    'flag_work_phone': 'bool',
    'flag_phone'     : 'bool',
    'flag_email'     : 'bool',


    # Category columns (keeping only those that make sense as categories)
    'admission_type'      : 'category',      # VARCHAR(40) NOT NULL
    'transaction_type'    : 'category',      # VARCHAR - POE
    'eventtype'           : 'category',      # VARCHAR - transfers (limited values)
    'order_status'        : 'category',      # VARCHAR - poe (limited status values)
    'interpretation'      : 'category',      # VARCHAR - microbiologyevents
    'sliding_scale'       : 'category',      # VARCHAR - pharmacy (limited values)
    'order_type'          : 'category',      # VARCHAR - poe (limited types)
    'priority'            : 'category',      # VARCHAR - labevents (limited priority levels)
    }

_CATEGORIES = {
    'admission_type': ['AMBULATORY OBSERVATION', 'DIRECT EMER.', 'DIRECT OBSERVATION', 'ELECTIVE', 'EU OBSERVATION', 'EW EMER.', 'OBSERVATION ADMIT', 'SURGICAL SAME DAY ADMISSION', 'URGENT' ],

    'transaction_type': ['Change', 'Co', 'D/C', 'H', 'New', 'T'], # The action which the provider performed when performing this order.

    'event_type': ['ed', 'admit', 'transfer', 'discharge'], # describes what transfer event occurred: ‘ed’ for an emergency department stay, ‘admit’ for an admission to the hospital, ‘transfer’ for an intra-hospital transfer and ‘discharge’ for a discharge from the hospital.

    'order_status': ['Active', 'Inactive'],

    'interpretation': ['I', 'R', 'S', 'P'], # interpretation of the antibiotic sensitivity, and indicates the results of the test. “S” is sensitive, “R” is resistant, “I” is intermediate, and “P” is pending.

    'sliding_scale': ['Y', 'N'], # Indicates whether the medication should be given on a sliding scale: either ‘Y’ or ‘N’.

    'order_type': [ 'ADT orders', 'Blood Bank', 'Cardiology', 'Consults', 'Critical Care', 'General Care', 'Hemodialysis', 'IV therapy', 'Lab', 'Medications', 'Neurology', 'Nutrition', 'OB', 'Radiology', 'Respiratory', 'TPN' ],
    'priority': [ 'ROUTINE', 'STAT']
    }

_DATETIME_COLUMNS = [
    'admittime',
    'chartdate',
    'charttime',
    'deathtime',
    'dischtime',
    'dod',
    'edouttime',
    'edregtime',
    'entertime',
    'expirationdate',
    'intime',
    'ordertime',
    'outtime',
    'scheduletime',
    'starttime',
    'stoptime',
    'storedate',
    'storetime',
    'transfertime',
    'verifiedtime'
    ]

_TABLES_W_SUBJECT_ID_COLUMN =  {'merged_table', 'patients', 'admissions', 'transfers', 'diagnoses_icd', 'poe', 'poe_detail', 'microbiologyevents', 'labevents', 'prescriptions'}


# Attach constants to the TableNames class
TableNames._HOSP_TABLES                = _HOSP_TABLES
TableNames._ICU_TABLES                 = _ICU_TABLES
TableNames._COLUMN_TO_TABLES           = _COLUMN_TO_TABLES
TableNames._TABLE_TO_COLUMNS           = _TABLE_TO_COLUMNS
TableNames._COLUMN_TYPES               = _COLUMN_TYPES
TableNames._CATEGORIES                 = _CATEGORIES
TableNames._DATETIME_COLUMNS           = _DATETIME_COLUMNS
TableNames._TABLES_W_SUBJECT_ID_COLUMN = _TABLES_W_SUBJECT_ID_COLUMN

def values(cls) -> list[str]:
    """Returns all table values as a list."""
    return [member.value for member in cls]

def description(self) -> str:
    """Returns a description of the table."""
    descriptions = {
        # Merged Tables
        'merged_table': "Merged table combining relevant columns from multiple HOSP and ICU sources",

        # HOSP Module Tables
        'admissions'        : "Patient hospital admissions information",
        'patients'          : "Patient demographic data",
        'transfers'         : "Patient transfer events within the hospital",
        'diagnoses_icd'     : "ICD diagnosis codes for patients",
        'procedures_icd'    : "ICD procedure codes for patients",
        'labevents'         : "Laboratory test results",
        'prescriptions'     : "Medication prescriptions",
        'microbiologyevents': "Microbiology test results",
        'pharmacy'          : "Pharmacy medication administration records",
        'poe'               : "Provider order entry records",
        'poe_detail'        : "Detailed provider order entry information",
        'hcpcsevents'       : "HCPCS procedure codes",
        'drgcodes'          : "Diagnosis-related group codes",
        'emar'              : "Electronic medication administration records",
        'emar_detail'       : "Detailed electronic medication administration records",
        'services'          : "Patient service assignments",
        'provider'          : "Healthcare provider information",
        'd_hcpcs'           : "Dictionary of HCPCS codes",
        'd_icd_diagnoses'   : "Dictionary of ICD diagnosis codes",
        'd_icd_procedures'  : "Dictionary of ICD procedure codes",
        'd_labitems'        : "Dictionary of laboratory items",
        'omr'               : "Outpatient medical records",

        # ICU Module Tables
        'caregiver'       : "ICU caregiver information",
        'chartevents'     : "ICU chart events and vital signs",
        'datetimeevents'  : "ICU datetime events",
        'd_items'         : "Dictionary of ICU items",
        'icustays'        : "ICU stay information",
        'ingredientevents': "ICU ingredient events",
        'inputevents'     : "ICU input events (fluids, medications)",
        'outputevents'    : "ICU output events (urine, drainage)",
        'procedureevents' : "ICU procedure events"
    }
    return descriptions.get(self.value, f"Table: {self.value}")

def module(self) -> str:
    """Returns the module (hosp or icu) that this table belongs to."""
    if self == TableNames.MERGED:
        return "merged_table"
    elif self.value in self.__class__._HOSP_TABLES:
        return "hosp"
    elif self.value in self.__class__._ICU_TABLES:
        return "icu"
    else:
        return "unknown"

def get_tables_with_column(cls, column_name: str) -> list[str]:
    """
    Get all tables that contain the specified column.

    Example:
        >>> TableNames.get_tables_with_column('subject_id')
        ['admissions', 'patients', 'transfers', ...]
    """
    return cls._COLUMN_TO_TABLES.get(column_name, [])

def columns(self) -> list[str]:
    """Returns all columns for the specified table."""
    return self.__class__._TABLE_TO_COLUMNS.get(self.value, [])

def types(self) -> str:
    """Returns the data type of a specific column within the table."""
    return {k:v for k,v in self.__class__._COLUMN_TYPES.items() if k in self.columns}

# Bind methods to the TableNames class
TableNames.description            = property(description)
TableNames.module                 = property(module)
TableNames.columns                = property(columns)
TableNames.types                  = property(types)
TableNames.values                 = classmethod(values)
TableNames.get_tables_with_column = classmethod(get_tables_with_column)



# ============================================================================
# Default study tables
# ============================================================================
DEFAULT_STUDY_TABLES_LIST = [
				TableNames.PATIENTS.value,  # 2.8MB
				TableNames.ADMISSIONS.value, # 19.9MB
				TableNames.DIAGNOSES_ICD.value, # 33.6MB
				TableNames.TRANSFERS.value, # 46MB
				TableNames.D_ICD_DIAGNOSES.value, # 876KB
				TableNames.POE.value, # 606MB
				TableNames.POE_DETAIL.value, # 55MB
                TableNames.PRESCRIPTIONS.value, # 606MB
                # TableNames.D_LABITEMS.value # 30KB
                # TableNames.LABEVENTS.value, # 2.6GB
			]


# ============================================================================
# Default paths and configurations
# ============================================================================
# TODO: Remove all dependencies on DEFAULT_MIMIC_PATH path except for UI path selection default value.
# DEFAULT_MIMIC_PATH = Path("/Users/artinmajdi/Library/CloudStorage/GoogleDrive-msm2024@gmail.com/My Drive/MIMIC_IV_Dataset")
DEFAULT_MIMIC_PATH = Path("/Users/pankajvyas/MIMIC_Project/MIMIC-IV-Raw Data")


DEFAULT_NUM_SUBJECTS = 20
RANDOM_STATE         = 42
SUBJECT_ID_COL       = 'subject_id'


# ============================================================================
# Type definitions
# ============================================================================
DataFrameType = pd.DataFrame | dd.DataFrame

# ============================================================================
# PyArrow type mappings
# ============================================================================
pyarrow_dtypes_map = {
	'int64'         : pa.int64(),
    'int32'         : pa.int32(),
	'int16'         : pa.int16(),  # Added for hospital_expire_flag and other SMALLINT columns
    'Int64'         : pa.int64(),
    'Int32'         : pa.int32(),
    'Int16'         : pa.int16(),
	'float64'       : pa.float64(),
	'bool'          : pa.bool_(),
	'datetime64[ns]': pa.timestamp('ns'),
	'category'      : pa.dictionary(pa.int32(), pa.string()),
	'object'        : pa.string(),
	'string'        : pa.string()
}
