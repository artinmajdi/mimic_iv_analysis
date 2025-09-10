# Standard library imports
import logging
from typing import Tuple

# Data processing imports
import numpy as np
import pandas as pd
import dask.dataframe as dd

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go

# Machine learning imports
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Streamlit import
import streamlit as st

# Local application imports
from mimic_iv_analysis.core.clustering import ClusterInterpreter


class AnalysisVisualizationTab:
	""" Handles the UI and logic for the post-clustering Analysis & Visualization tab. """

	def __init__(self):
		self.cluster_interpreter = ClusterInterpreter()

	def render(self):
		""" Renders the content of the Analysis & Visualization tab. """

		st.markdown("<h2 class='sub-header'>Cluster Analysis & Interpretation</h2>", unsafe_allow_html=True)

		# Introductory text
		st.markdown("""
		<div class='info-box'>
		Explore and interpret the identified clusters. Analyze differences in patient characteristics, visualize patterns, and generate reports to understand the meaning behind the groupings.
		</div>
		""", unsafe_allow_html=True)

		# --- Select Clustering Result ---
		available_labels = {}
		if 'kmeans_labels' in st.session_state and st.session_state.kmeans_labels is not None:
			available_labels['K-means'] = st.session_state.kmeans_labels

		if 'hierarchical_labels' in st.session_state and st.session_state.hierarchical_labels is not None:
			available_labels['Hierarchical (Sampled)'] = st.session_state.hierarchical_labels # Note it's sampled

		if 'dbscan_labels' in st.session_state and st.session_state.dbscan_labels is not None:
			available_labels['DBSCAN'] = st.session_state.dbscan_labels

		# Add LDA dominant topic if available
		if 'lda_results' in st.session_state and st.session_state.lda_results:
			doc_topic_df = st.session_state.lda_results['doc_topic_matrix']
			if not doc_topic_df.empty:
				available_labels['LDA Dominant Topic'] = doc_topic_df.idxmax(axis=1)

		if not available_labels:
			st.warning("No clustering or topic modeling results found in the current session. Please run an algorithm in the 'Clustering Analysis' tab first.")
			return

		selected_clustering_name = st.selectbox( label="Select Clustering/Topic Result to Analyze", options=list(available_labels.keys()) )

		cluster_labels = available_labels[selected_clustering_name]

		# --- Get Data for Analysis ---
		# Prefer original preprocessed data for interpretation if possible
		analysis_data = None
		if 'clustering_input_data' in st.session_state and st.session_state.clustering_input_data is not None:
			analysis_data = st.session_state.clustering_input_data.copy()
			# Align data with labels (important if sampling occurred, e.g., hierarchical)
			common_index = analysis_data.index.intersection(cluster_labels.index)
			if len(common_index) != len(cluster_labels):
				st.warning(f"Analysis data index ({len(analysis_data)}) does not fully match cluster label index ({len(cluster_labels)}). Analyzing subset with {len(common_index)} common points.")
			analysis_data = analysis_data.loc[common_index]
			cluster_labels = cluster_labels.loc[common_index]

		elif 'df' in st.session_state and st.session_state.df is not None:
			# Fallback to original df if preprocessed is missing, but warn user
			analysis_data  = st.session_state.df.copy()
			common_index   = analysis_data.index.intersection(cluster_labels.index)
			analysis_data  = analysis_data.loc[common_index]
			cluster_labels = cluster_labels.loc[common_index]

			st.warning("Using original loaded table data for analysis as preprocessed clustering input is unavailable. Results might be less meaningful if data wasn't scaled/numeric.")

			# Try to select only numeric columns for some analyses
			analysis_data_numeric = analysis_data.select_dtypes(include=np.number)
			if not analysis_data_numeric.empty:
				analysis_data = analysis_data_numeric
			else:
				st.error("No numeric data available in the original table for analysis.")
				analysis_data = None # Cannot proceed

		else:
			st.error("No suitable data found for cluster analysis (neither preprocessed input nor original table).")
			return # Cannot proceed


		if analysis_data is not None and not analysis_data.empty:
			# Add cluster labels to the analysis data
			analysis_data['cluster'] = cluster_labels.astype(str) # Use string for categorical coloring/grouping
			# Handle DBSCAN noise label
			if selected_clustering_name == 'DBSCAN':
				analysis_data['cluster'] = analysis_data['cluster'].replace('-1', 'Noise')

			# Filter out noise for some analyses if needed
			analysis_data_no_noise = analysis_data[analysis_data['cluster'] != 'Noise'] if 'Noise' in analysis_data['cluster'].unique() else analysis_data

			# --- Analysis Tabs ---
			analysis_tabs = st.tabs([
				"📊 Cluster Profiles",
				"🔍 Statistical Differences",
				"🔥 Feature Importance",
				"📈 LOS / Outcome Analysis",
				"📋 Generate Report"
			])

			# 1. Cluster Profiles Tab
			with analysis_tabs[0]:
				st.markdown("<h3>Cluster Profiles & Characteristics</h3>", unsafe_allow_html=True)
				st.info("Explore the average characteristics of each cluster based on the input features.")

				try:
					# Calculate summary statistics per cluster (using data without noise for means)
					cluster_summary = analysis_data_no_noise.groupby('cluster').agg(['mean', 'median', 'std', 'count'])

					# Display summary table
					st.dataframe(cluster_summary.style.format("{:.3f}", na_rep="-"), use_container_width=True)

					# Select features for radar plot
					profile_features = st.multiselect(
						"Select features for Radar Plot Profile",
						[col for col in analysis_data_no_noise.columns if col != 'cluster'],
						default=[col for col in analysis_data_no_noise.columns if col != 'cluster'][:min(8, analysis_data_no_noise.shape[1]-1)] # Default up to 8 features
					)

					if profile_features:
						# Normalize data for radar plot (e.g., Min-Max scaling across all data)
						scaler = MinMaxScaler()
						radar_data = analysis_data_no_noise[profile_features].copy()
						# Handle potential NaNs before scaling
						radar_data = radar_data.fillna(radar_data.median()) # Impute with median
						scaled_values = scaler.fit_transform(radar_data)
						scaled_df = pd.DataFrame(scaled_values, columns=profile_features, index=radar_data.index)
						scaled_df['cluster'] = analysis_data_no_noise['cluster']

						# Calculate mean scaled values per cluster
						radar_means = scaled_df.groupby('cluster')[profile_features].mean()

						# Create radar plot
						fig_radar = go.Figure()
						categories = radar_means.columns.tolist()

						for i, cluster_id in enumerate(radar_means.index):
							fig_radar.add_trace(go.Scatterpolar(
								r=radar_means.loc[cluster_id].values.flatten(),
								theta=categories,
								fill='toself',
								name=f"Cluster {cluster_id}",
								line=dict(color=px.colors.qualitative.G10[i % len(px.colors.qualitative.G10)])
							))

						fig_radar.update_layout(
							polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
							showlegend=True,
							title="Cluster Profiles (Scaled Feature Means)"
						)
						st.plotly_chart(fig_radar, use_container_width=True)
					else:
						st.info("Select features to generate the radar plot.")

				except Exception as e:
					st.error(f"Error generating cluster profiles: {e}")
					logging.exception("Error in Cluster Profiles tab")


			# 2. Statistical Differences Tab
			with analysis_tabs[1]:
				st.markdown("<h3>Statistical Differences Between Clusters</h3>", unsafe_allow_html=True)
				st.info("Identify features that show statistically significant differences across the clusters (excluding noise points).")

				if len(analysis_data_no_noise['cluster'].unique()) < 2:
					st.warning("Need at least two non-noise clusters to perform statistical tests.")
				else:
					# Get feature columns (exclude cluster column)
					feature_cols = [col for col in analysis_data_no_noise.columns if col != 'cluster']

					# Let user select features
					selected_features_test = st.multiselect(
						"Select Features for Statistical Testing",
						feature_cols,
						default=feature_cols[:min(10, len(feature_cols))], # Default up to 10
						key="stat_test_features"
					)

					# Test method based on number of clusters
					num_clusters_no_noise = len(analysis_data_no_noise['cluster'].unique())
					if num_clusters_no_noise == 2:
						test_method = st.radio("Statistical Test Method", ["t-test (parametric)", "Mann-Whitney U (non-parametric)"], horizontal=True, key="stat_test_method_2")
						method_map = {"t-test (parametric)": "ttest", "Mann-Whitney U (non-parametric)": "mannwhitneyu"}
					else: # > 2 clusters
						test_method = st.radio("Statistical Test Method", ["ANOVA (parametric)", "Kruskal-Wallis (non-parametric)"], horizontal=True, key="stat_test_method_multi")
						method_map = {"ANOVA (parametric)": "anova", "Kruskal-Wallis (non-parametric)": "kruskal"}


					# Button to run tests
					if st.button("Run Statistical Tests") and selected_features_test:
						try:
							with st.spinner("Performing statistical tests..."):
								# Run tests using the analyzer
								test_results = self.cluster_interpreter.statistical_testing(
									analysis_data_no_noise, # Use data without noise
									selected_features_test,
									cluster_col='cluster',
									method=method_map[test_method]
								)

							st.success("Statistical tests completed!")
							st.dataframe(test_results.style.format({
								'Statistic': '{:.3f}',
								'P-Value': '{:.4g}', # General format for p-values
								'Adjusted P-Value': '{:.4g}'
							}).applymap(
								lambda v: 'background-color: lightcoral' if isinstance(v, bool) and v else '', subset=['Significant (Adjusted)']
							), use_container_width=True)

							# --- P-value Visualization ---
							st.markdown("<h4>Feature Significance Visualization (-log10 Adjusted P-Value)</h4>", unsafe_allow_html=True)
							results_vis = test_results.dropna(subset=['Adjusted P-Value']).copy() # Drop features where test failed
							if not results_vis.empty:
								# Avoid log(0) errors
								results_vis['log_p'] = -np.log10(results_vis['Adjusted P-Value'] + 1e-10) # Add small epsilon
								significance_threshold = -np.log10(0.05)

								fig_pvals = px.bar(
									results_vis.sort_values('log_p', ascending=False),
									x='Feature', y='log_p',
									color='Significant (Adjusted)',
									color_discrete_map={True: 'red', False: 'grey'},
									labels={'log_p': '-log10(Adjusted P-Value)'},
									title='Feature Significance by Adjusted P-Value'
								)
								fig_pvals.add_hline(y=significance_threshold, line_dash="dash", annotation_text="p=0.05 Threshold")
								st.plotly_chart(fig_pvals, use_container_width=True)
							else:
								st.info("No valid adjusted p-values to visualize.")

						except AttributeError:
							st.error("Cluster Analyzer is not properly initialized or does not have a 'statistical_testing' method.")
						except Exception as e:
							st.error(f"Error performing statistical tests: {str(e)}")
							logging.exception("Error in Statistical Testing tab")
					elif not selected_features_test:
						st.warning("Please select features to run statistical tests.")


			# 3. Feature Importance Tab
			with analysis_tabs[2]:
				st.markdown("<h3>Feature Importance for Cluster Separation</h3>", unsafe_allow_html=True)
				st.info("Identify features that are most important for distinguishing between the clusters using a Random Forest classifier (trained on clusters vs features).")

				if len(analysis_data_no_noise['cluster'].unique()) < 2:
					st.warning("Need at least two non-noise clusters to calculate feature importance.")
				else:
					# Button to calculate importance
					if st.button("Calculate Feature Importance (using RandomForest)"):
						try:
							with st.spinner("Calculating feature importance..."):
								# Calculate importance using the analyzer
								importance_df = self.cluster_interpreter.calculate_feature_importance(
									analysis_data_no_noise, # Use data without noise
									cluster_col='cluster'
								)

							st.success("Feature importance calculated!")
							st.dataframe(importance_df, use_container_width=True)

							# --- Importance Visualization ---
							fig_imp = px.bar(
								importance_df.head(20).sort_values('Importance', ascending=True), # Show top 20
								x='Importance', y='Feature', orientation='h',
								title='Top 20 Most Important Features for Cluster Separation'
							)
							st.plotly_chart(fig_imp, use_container_width=True)

						except ImportError as e:
							st.error(f"Missing dependency for feature importance: {e}. Please install scikit-learn (`pip install scikit-learn`).")
						except AttributeError:
							st.error("Cluster Analyzer is not properly initialized or does not have a 'calculate_feature_importance' method.")
						except Exception as e:
							st.error(f"Error calculating feature importance: {str(e)}")
							logging.exception("Error in Feature Importance tab")


			# 4. LOS / Outcome Analysis Tab
			with analysis_tabs[3]:

				def _find_indices_intersection(df: pd.DataFrame, cluster_labels: pd.Series) -> Tuple[pd.Index, pd.Index]:
        
					# Handle index alignment for both Dask and pandas DataFrames
					original_index = df.index.compute() if isinstance(df, dd.DataFrame) else df.index

					# Find common indices between original data and cluster labels
					common_index_outcome = original_index.intersection(cluster_labels.index)
					
					if len(common_index_outcome) > 0:
						# Use the common indices to filter both DataFrames
						original_df_aligned    = df.loc[common_index_outcome]
						cluster_labels_aligned = cluster_labels.loc[common_index_outcome]
						return original_df_aligned, cluster_labels_aligned, common_index_outcome

					st.error("No common indices found between original data and cluster labels.")
					return pd.Index([]), pd.Index([]), pd.Index([])


				st.markdown("<h3>Length of Stay (LOS) or Outcome Analysis by Cluster</h3>", unsafe_allow_html=True)
				st.info("Compare clinical outcomes like Length of Stay across the identified clusters. Requires appropriate columns in the *original* loaded table.")

				if st.session_state.df is None:
					st.warning("Original table data not loaded. Cannot perform outcome analysis.")
				else:
					original_df = st.session_state.df

					try:
						original_df_aligned, cluster_labels_aligned, common_index_outcome = _find_indices_intersection(df=original_df, cluster_labels=cluster_labels)

					except Exception as e:
						st.error(f"Error aligning Dask DataFrame indices: {str(e)}")
						st.warning("Fallback: Using sample data for outcome analysis due to index alignment issues.")

						# Fallback: compute a sample of the original DataFrame
						try:
							original_sample = original_df.head(min(1000, len(cluster_labels)), compute=True)
							original_df_aligned, cluster_labels_aligned, common_index_outcome = _find_indices_intersection(df=original_sample, cluster_labels=cluster_labels)
        
						except Exception as fallback_error:
							st.error(f"Fallback failed: {str(fallback_error)}")
							common_index_outcome = pd.Index([])


					if len(common_index_outcome) == 0:
						st.error("Index mismatch between original data and cluster labels. Cannot perform outcome analysis.")
					else:

						# --- LOS Calculation ---
						st.markdown("#### Length of Stay (LOS)")
						time_columns = original_df_aligned.select_dtypes(include=['datetime64[ns]', 'datetime64[us]', 'datetime64[ms]']).columns.tolist()
						# Try to find potential date columns from objects/strings if no datetime found
						if not time_columns:
							object_cols = original_df_aligned.select_dtypes(include=['object', 'string']).columns
							for col in object_cols:
								try:
									# Attempt conversion on a sample - very basic check
									pd.to_datetime(original_df_aligned[col].dropna().iloc[:5], errors='raise')
									time_columns.append(col)
								except (ValueError, TypeError, AttributeError, IndexError):
									continue # Cannot convert reliably

						if not time_columns:
							st.warning("No datetime columns found or detected in the original table for LOS calculation.")
						else:
							col1, col2, col3 = st.columns(3)
							with col1:
								# Try to guess admission column
								admit_guess   = [c for c in time_columns if 'admit' in c.lower()]
								admit_idx     = time_columns.index(admit_guess[0]) if admit_guess else 0
								admission_col = st.selectbox("Admission Time Column", time_columns, index=admit_idx, key="los_admit")
							with col2:
								# Try to guess discharge column
								disch_guess   = [c for c in time_columns if 'disch' in c.lower()]
								disch_idx     = time_columns.index(disch_guess[0]) if disch_guess else (1 if len(time_columns)>1 else 0)
								discharge_col = st.selectbox("Discharge Time Column", time_columns, index=disch_idx, key="los_disch")
							with col3:
								# Try to guess patient ID
								id_guess       = [c for c in original_df_aligned.columns if 'subject_id' in c.lower() or 'patient_id' in c.lower()]
								id_idx         = original_df_aligned.columns.tolist().index(id_guess[0]) if id_guess else 0
								patient_id_col = st.selectbox("Patient ID Column (for grouping)", original_df_aligned.columns.tolist(), index=id_idx, key="los_id")


							if st.button("Analyze Length of Stay by Cluster"):
								try:
									with st.spinner("Calculating LOS and comparing across clusters..."):
										# Calculate LOS using analyzer
										los_data = self.cluster_interpreter.calculate_length_of_stay(
											original_df_aligned, # Use aligned data
											admission_col,
											discharge_col,
											patient_id_col
										)

										# Add cluster labels (aligned)
										los_data_clustered = los_data.to_frame(name='los_days').join(cluster_labels_aligned.to_frame(name='cluster'))
										los_data_clustered = los_data_clustered.dropna() # Drop patients where LOS or cluster is missing

										# Add DBSCAN noise handling
										if selected_clustering_name == 'DBSCAN':
											los_data_clustered['cluster'] = los_data_clustered['cluster'].replace('-1', 'Noise')

										# Display summary stats
										st.markdown("##### LOS Summary Statistics by Cluster")
										los_summary = los_data_clustered.groupby('cluster')['los_days'].agg(['mean', 'median', 'std', 'count'])
										st.dataframe(los_summary.style.format("{:.2f}"), use_container_width=True)

										# --- LOS Visualization ---
										st.markdown("##### LOS Distribution by Cluster")
										fig_los = px.box(los_data_clustered, x='cluster', y='los_days',
														title="Length of Stay Distribution by Cluster",
														labels={'los_days': 'Length of Stay (Days)'},
														points='outliers', color='cluster')
										st.plotly_chart(fig_los, use_container_width=True)

										# --- Statistical Test for LOS ---
										st.markdown("##### Statistical Test for LOS Differences")
										los_no_noise = los_data_clustered[los_data_clustered['cluster'] != 'Noise']
										unique_clusters = los_no_noise['cluster'].unique()

										if len(unique_clusters) < 2:
											st.info("Need at least two non-noise clusters for statistical comparison.")
										elif len(unique_clusters) == 2:
											# Mann-Whitney U test (non-parametric often safer for LOS)
											group1 = los_no_noise[los_no_noise['cluster'] == unique_clusters[0]]['los_days']
											group2 = los_no_noise[los_no_noise['cluster'] == unique_clusters[1]]['los_days']
											stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
											st.markdown(f"**Mann-Whitney U Test:** Statistic={stat:.3f}, p-value={p_val:.4g}")

											if p_val < 0.05:
												st.success("Significant difference in LOS found (p < 0.05).")
											else:
												st.info("No significant difference in LOS found (p >= 0.05).")
										else:
											# Kruskal-Wallis test (non-parametric ANOVA)
											groups = [los_no_noise[los_no_noise['cluster'] == c]['los_days'] for c in unique_clusters]
											stat, p_val = stats.kruskal(*groups)
											st.markdown(f"**Kruskal-Wallis Test:** Statistic={stat:.3f}, p-value={p_val:.4g}")
											if p_val < 0.05:
												st.success("Significant difference in LOS found across clusters (p < 0.05).")
											else:
												st.info("No significant difference in LOS found across clusters (p >= 0.05).")

								except AttributeError:
									st.error("Cluster Analyzer is not properly initialized or does not have a 'calculate_length_of_stay' method.")
								except KeyError as e:
									st.error(f"Column Error: Could not find column '{e}'. Check selections.")
								except ValueError as e:
									st.error(f"Data Error: {e}. Ensure time columns are in a parsable format and admit time is before discharge time.")
								except Exception as e:
									st.error(f"Error analyzing LOS: {str(e)}")
									logging.exception("Error in LOS Analysis")


						# --- Other Outcome Analysis (Example: Mortality) ---
						st.markdown("#### Other Outcome Analysis (e.g., Mortality)")
						# Find potential mortality columns
						mortality_cols = [c for c in original_df_aligned.columns if 'mortality' in c.lower() or 'death' in c.lower() or 'expire_flag' in c.lower()]
						if mortality_cols:
							outcome_col = st.selectbox("Select Outcome Column (e.g., Mortality Flag)", mortality_cols + ["None"], index=0)
							if outcome_col != "None":
								if st.button(f"Analyze {outcome_col} by Cluster"):
									try:
										# Ensure outcome is binary (0/1) or boolean
										outcome_data = original_df_aligned[[outcome_col]].join(cluster_labels_aligned.to_frame(name='cluster'))
										outcome_data = outcome_data.dropna()
										# Attempt conversion to numeric/boolean
										try:
											outcome_data[outcome_col] = pd.to_numeric(outcome_data[outcome_col])
											# Check if mostly 0s and 1s
											if not outcome_data[outcome_col].isin([0, 1]).all():
												st.warning(f"Outcome column '{outcome_col}' contains values other than 0/1. Analysis might be invalid. Trying anyway...")
										except ValueError:
											st.warning(f"Could not convert outcome column '{outcome_col}' to numeric. Trying boolean conversion.")
											outcome_data[outcome_col] = outcome_data[outcome_col].astype(bool)


										# Add DBSCAN noise handling
										if selected_clustering_name == 'DBSCAN':
											outcome_data['cluster'] = outcome_data['cluster'].replace('-1', 'Noise')

										st.markdown(f"##### {outcome_col} Rate by Cluster")
										# Calculate rate (assuming 1 = event, 0 = no event)
										outcome_summary = outcome_data.groupby('cluster')[outcome_col].agg(['mean', 'count'])
										outcome_summary.rename(columns={'mean': 'Event Rate'}, inplace=True)
										st.dataframe(outcome_summary.style.format({'Event Rate': '{:.1%}'}), use_container_width=True) # Format as percentage

										# --- Chi-squared Test ---
										st.markdown(f"##### Statistical Test for {outcome_col} Differences")
										outcome_no_noise = outcome_data[outcome_data['cluster'] != 'Noise']
										if len(outcome_no_noise['cluster'].unique()) >= 2:
											contingency_table = pd.crosstab(outcome_no_noise['cluster'], outcome_no_noise[outcome_col])
											st.write("Contingency Table (Cluster vs Outcome):")
											st.dataframe(contingency_table)
											chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
											st.markdown(f"**Chi-squared Test:** Chi2={chi2:.3f}, p-value={p_val:.4g}")
											if p_val < 0.05: st.success(f"Significant difference in {outcome_col} found across clusters (p < 0.05).")
											else: st.info(f"No significant difference in {outcome_col} found across clusters (p >= 0.05).")
										else:
											st.info("Need at least two non-noise clusters for Chi-squared test.")

									except Exception as e:
										st.error(f"Error analyzing outcome '{outcome_col}': {e}")
										logging.exception(f"Error analyzing outcome {outcome_col}")
						else:
							st.info("No columns matching typical mortality indicators found in the original table.")


			# 5. Generate Report Tab
			with analysis_tabs[4]:
				st.markdown("<h3>Generate Cluster Analysis Report</h3>", unsafe_allow_html=True)
				st.info("Create a downloadable HTML report summarizing the cluster analysis findings, including profiles, statistics, and visualizations.")

				report_title = st.text_input("Report Title", value=f"{selected_clustering_name} Analysis Report")
				include_plots_report = st.checkbox("Include Visualizations in Report", value=True)

				# Add option to select which sections to include
				st.markdown("Select sections to include:")
				include_profile = st.checkbox("Cluster Profiles", value=True)
				include_stats = st.checkbox("Statistical Differences", value=True)
				include_importance = st.checkbox("Feature Importance", value=True)
				include_outcome = st.checkbox("LOS/Outcome Analysis", value=True)
				# Add more sections as needed

				if st.button("Generate HTML Report"):
					try:
						with st.spinner("Generating report..."):
							# Gather data for the report (this might involve re-running parts of the analysis or retrieving from session state)
							report_data = {
								'title': report_title,
								'clustering_method': selected_clustering_name,
								'analysis_data': analysis_data, # Pass the data used
								'cluster_labels': cluster_labels, # Pass the labels
								'include_plots': include_plots_report,
								'sections': {
									'profile': include_profile,
									'stats': include_stats,
									'importance': include_importance,
									'outcome': include_outcome
								},
								# Add other necessary data like test results, importance scores, LOS data etc.
								# These might need to be explicitly passed or retrieved from session state
								# Example:
								# 'stat_test_results': st.session_state.get('last_stat_test_results'),
								# 'feature_importance': st.session_state.get('last_feature_importance'),
								# 'los_data': st.session_state.get('last_los_data_clustered'),
							}

							# Generate HTML using the analyzer's method
							# This method needs to be implemented in MIMICClusterAnalyzer
							html_content = self.cluster_interpreter.generate_html_report(report_data)

						st.success("Report generated successfully!")
						st.download_button(
							label="Download HTML Report",
							data=html_content,
							file_name=f"{selected_clustering_name}_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
							mime="text/html",
						)
					except NotImplementedError:
						st.error("Report generation functionality is not yet implemented in the Cluster Analyzer.")
					except AttributeError:
						st.error("Cluster Analyzer is not properly initialized or does not have a 'generate_html_report' method.")
					except Exception as e:
						st.error(f"Error generating report: {str(e)}")
						logging.exception("Error generating report")

