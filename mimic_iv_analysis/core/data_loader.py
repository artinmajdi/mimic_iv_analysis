# Standard library imports
import os
import glob
import logging
from typing import Dict, Optional, Tuple

# Data processing imports
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Constants
DEFAULT_MIMIddC_PATH    = "/Users/artinmajdi/Documents/GitHubs/Career/mimic_iv/dataset/mimic-iv-3.1"
LARGE_FILE_THRESHOLD_MB = 100
DEFAULT_SAMPLE_SIZE     = 1000
RANDOM_STATE            = 42

class MIMICDataLoader:
	"""Handles scanning, loading, and providing info for MIMIC-IV data."""

	def __init__(self):
		pass


	def _format_size(self, size_mb: float) -> str:
		"""Format file size into KB, MB, or GB string."""
		if size_mb < 0.001:
			return "(< 1 KB)"
		elif size_mb < 1:
			return f"({size_mb * 1024:.1f} KB)"
		elif size_mb < 1000:
			return f"({size_mb:.1f} MB)"
		else:
			return f"({size_mb / 1000:.1f} GB)"


	def scan_mimic_directory(self, mimic_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
		"""Scans the MIMIC-IV directory structure and returns table info."""
		available_tables    = {}
		file_paths          = {}
		file_sizes          = {}
		table_display_names = {}

		if not os.path.exists(mimic_path):
			return available_tables, file_paths, file_sizes, table_display_names

		modules = ['hosp', 'icu']
		for module in modules:
			module_path = os.path.join(mimic_path, module)
			if os.path.exists(module_path):
				available_tables[module] = []
				current_module_tables = {}

				csv_files = glob.glob(os.path.join(module_path, '*.csv'))
				csv_gz_files = glob.glob(os.path.join(module_path, '*.csv.gz'))

				all_files = csv_files + csv_gz_files

				for file_path in all_files:
					is_gz = file_path.endswith('.csv.gz')
					table_name = os.path.basename(file_path).replace('.csv.gz' if is_gz else '.csv', '')

					if table_name in current_module_tables and not is_gz:
						current_module_tables[table_name] = file_path

					elif table_name not in current_module_tables:
						current_module_tables[table_name] = file_path

				for table_name, file_path in current_module_tables.items():
					available_tables[module].append(table_name)
					file_paths[(module, table_name)] = file_path
					try:
						file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

					except OSError:
						file_size_mb = 0

					file_sizes[(module, table_name)] = file_size_mb
					size_str = self._format_size(file_size_mb)
					table_display_names[(module, table_name)] = f"{table_name} {size_str}"

				available_tables[module].sort()

		return available_tables, file_paths, file_sizes, table_display_names


	def load_mimic_table(self, file_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE, encoding: str = 'latin-1') -> Optional[pd.DataFrame]:
		"""Loads a specific MIMIC-IV table, handling large files and sampling."""
		try:
			file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
			is_compressed = file_path.endswith('.gz')
			compression = 'gzip' if is_compressed else None

			read_params = {
				'encoding': encoding,
				'compression': compression,
				'low_memory': False
			}

			if file_size_mb > LARGE_FILE_THRESHOLD_MB:
				df = pd.read_csv(file_path, nrows=sample_size, **read_params)
			else:
				df = pd.read_csv(file_path, **read_params)
				if len(df) > sample_size:
					df = df.sample(sample_size, random_state=RANDOM_STATE)
			return df
		except Exception as e:
			logging.error(f"Error loading data from {os.path.basename(file_path)}: {str(e)}")
			return None


	def get_table_info(self, module: str, table_name: str) -> str:
		"""Get descriptive information about a specific MIMIC-IV table."""
		table_info = {
			('hosp', 'admissions'): "Patient hospital admissions information",
			('hosp', 'patients'): "Patient demographic data",
			('hosp', 'labevents'): "Laboratory measurements (large file)",
			('hosp', 'microbiologyevents'): "Microbiology test results",
			('hosp', 'pharmacy'): "Pharmacy orders",
			('hosp', 'prescriptions'): "Medication prescriptions",
			('hosp', 'procedures_icd'): "Patient procedures",
			('hosp', 'diagnoses_icd'): "Patient diagnoses",
			('hosp', 'emar'): "Electronic medication administration records",
			('hosp', 'emar_detail'): "Detailed medication administration data",
			('hosp', 'poe'): "Provider order entries",
			('hosp', 'poe_detail'): "Detailed order information",
			('hosp', 'd_hcpcs'): "HCPCS code definitions",
			('hosp', 'd_icd_diagnoses'): "ICD diagnosis code definitions",
			('hosp', 'd_icd_procedures'): "ICD procedure code definitions",
			('hosp', 'd_labitems'): "Laboratory test definitions",
			('hosp', 'hcpcsevents'): "HCPCS events",
			('hosp', 'drgcodes'): "Diagnosis-related group codes",
			('hosp', 'services'): "Hospital services",
			('hosp', 'transfers'): "Patient transfers",
			('hosp', 'provider'): "Provider information",
			('hosp', 'omr'): "Order monitoring results",
			('icu', 'chartevents'): "Patient charting data (vital signs, etc.)",
			('icu', 'datetimeevents'): "Date/time-based events",
			('icu', 'inputevents'): "Patient intake data",
			('icu', 'outputevents'): "Patient output data",
			('icu', 'procedureevents'): "ICU procedures",
			('icu', 'ingredientevents'): "Detailed medication ingredients",
			('icu', 'd_items'): "Dictionary of ICU items",
			('icu', 'icustays'): "ICU stay information",
			('icu', 'caregiver'): "Caregiver information"
		}
		return table_info.get((module, table_name), "No description available")


	def convert_to_parquet(self, df: pd.DataFrame, table_name: str, current_file_path: str) -> Optional[str]:
		"""Converts the loaded DataFrame to Parquet format. Returns path or None."""
		try:
			parquet_dir = os.path.join(os.path.dirname(current_file_path), 'parquet_files')
			os.makedirs(parquet_dir, exist_ok=True)
			parquet_file = os.path.join(parquet_dir, f"{table_name}.parquet")
			table = pa.Table.from_pandas(df)
			pq.write_table(table, parquet_file)
			return parquet_file
		except Exception as e:
			logging.error(f"Error converting {table_name} to Parquet: {str(e)}")
			return None

