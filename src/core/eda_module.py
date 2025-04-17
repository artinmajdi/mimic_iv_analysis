import streamlit as st
import pandas as pd
import plotly.express as px
import calendar


class ExploratoryDataAnalysis:
    """Class for exploratory data analysis of MIMIC-IV dataset."""

    def __init__(self, data_loader):
        """Initialize the EDA module with a data loader.

        Args:
            data_loader (MIMICDataLoader): Data loader object with preprocessed data
        """
        self.data_loader = data_loader
        self.data = data_loader.preprocessed

    def patient_demographics(self):
        """Display patient demographics and cohort statistics."""
        if 'patients' not in self.data or self.data['patients'] is None:
            st.warning("Patient data not available for demographic analysis.")
            return

        patients_df = self.data['patients']

        st.subheader("Patient Demographics and Cohort Statistics")

        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            if 'age' in patients_df.columns:
                st.write("#### Age Distribution")
                fig = px.histogram(
                    patients_df,
                    x='age',
                    nbins=20,
                    color_discrete_sequence=['#4682b4'],
                    title='Age Distribution'
                )
                fig.update_layout(
                    xaxis_title='Age',
                    yaxis_title='Count',
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age information not available in the dataset.")

            # Gender distribution
            if 'gender' in patients_df.columns:
                st.write("#### Gender Distribution")
                gender_counts = patients_df['gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']

                fig = px.pie(
                    gender_counts,
                    values='Count',
                    names='Gender',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Gender Distribution'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gender information not available in the dataset.")

        with col2:
            # Ethnicity distribution
            if 'race' in patients_df.columns:
                st.write("#### Ethnicity Distribution")

                # Get top 5 ethnicities and group others
                ethnicity_counts = patients_df['race'].value_counts()
                top_ethnicities = ethnicity_counts.nlargest(5).index.tolist()

                ethnicity_df = patients_df.copy()
                ethnicity_df['race_grouped'] = ethnicity_df['race'].apply(
                    lambda x: x if x in top_ethnicities else 'Other'
                )

                ethnicity_grouped = ethnicity_df['race_grouped'].value_counts().reset_index()
                ethnicity_grouped.columns = ['Ethnicity', 'Count']

                fig = px.bar(
                    ethnicity_grouped,
                    x='Ethnicity',
                    y='Count',
                    color='Ethnicity',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Ethnicity Distribution (Top 5)'
                )
                fig.update_layout(
                    xaxis_title='Ethnicity',
                    yaxis_title='Count',
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Ethnicity information not available in the dataset.")

            # Insurance distribution
            if 'admissions' in self.data and self.data['admissions'] is not None and 'insurance' in self.data['admissions'].columns:
                st.write("#### Insurance Distribution")

                insurance_counts = self.data['admissions']['insurance'].value_counts().reset_index()
                insurance_counts.columns = ['Insurance', 'Count']

                fig = px.pie(
                    insurance_counts,
                    values='Count',
                    names='Insurance',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Insurance Distribution'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insurance information not available in the dataset.")

        # Age and gender distribution
        if 'age' in patients_df.columns and 'gender' in patients_df.columns:
            st.write("#### Age Distribution by Gender")

            fig = px.histogram(
                patients_df,
                x='age',
                color='gender',
                nbins=20,
                barmode='overlay',
                opacity=0.7,
                color_discrete_sequence=['#4682b4', '#e377c2'],
                title='Age Distribution by Gender'
            )
            fig.update_layout(
                xaxis_title='Age',
                yaxis_title='Count',
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)

    def hospital_utilization(self):
        """Display hospital utilization patterns."""
        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for utilization analysis.")
            return

        admissions_df = self.data['admissions']

        st.subheader("Hospital Utilization Patterns")

        # Admission type distribution
        if 'admission_type' in admissions_df.columns:
            st.write("#### Admission Type Distribution")

            admission_type_counts = admissions_df['admission_type'].value_counts().reset_index()
            admission_type_counts.columns = ['Admission Type', 'Count']

            fig = px.bar(
                admission_type_counts,
                x='Admission Type',
                y='Count',
                color='Admission Type',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title='Admission Type Distribution'
            )
            fig.update_layout(
                xaxis_title='Admission Type',
                yaxis_title='Count',
                xaxis={'categoryorder':'total descending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Admission type information not available in the dataset.")

        # Department/service utilization
        if 'transfers' in self.data and self.data['transfers'] is not None and 'careunit' in self.data['transfers'].columns:
            st.write("#### Department/Care Unit Utilization")

            transfers_df = self.data['transfers']

            # Get top 10 care units by total patient days
            if 'unit_los_hours' in transfers_df.columns:
                unit_days = transfers_df.groupby('careunit')['unit_los_hours'].sum().div(24).reset_index()
                unit_days.columns = ['Care Unit', 'Total Patient Days']
                unit_days = unit_days.sort_values('Total Patient Days', ascending=False).head(10)

                fig = px.bar(
                    unit_days,
                    x='Care Unit',
                    y='Total Patient Days',
                    color='Care Unit',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Top 10 Care Units by Total Patient Days'
                )
                fig.update_layout(
                    xaxis_title='Care Unit',
                    yaxis_title='Total Patient Days',
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to simple count if LOS not available
                unit_counts = transfers_df['careunit'].value_counts().nlargest(10).reset_index()
                unit_counts.columns = ['Care Unit', 'Count']

                fig = px.bar(
                    unit_counts,
                    x='Care Unit',
                    y='Count',
                    color='Care Unit',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Top 10 Care Units by Patient Count'
                )
                fig.update_layout(
                    xaxis_title='Care Unit',
                    yaxis_title='Count',
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Care unit information not available in the dataset.")

        # Bed occupancy over time
        if 'admissions' in self.data and 'admittime' in admissions_df.columns and 'dischtime' in admissions_df.columns:
            st.write("#### Bed Occupancy Over Time")

            # Allow user to select time granularity
            time_granularity = st.selectbox(
                "Select Time Granularity",
                ["Daily", "Weekly", "Monthly"],
                index=2
            )

            # Get min and max dates
            min_date = admissions_df['admittime'].min().date()
            max_date = admissions_df['dischtime'].max().date()

            # Create date range based on granularity
            if time_granularity == "Daily":
                date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                date_format = '%Y-%m-%d'
            elif time_granularity == "Weekly":
                date_range = pd.date_range(start=min_date, end=max_date, freq='W')
                date_format = '%Y-%m-%d'
            else:  # Monthly
                date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
                date_format = '%Y-%m'

            # Initialize occupancy data
            occupancy_data = []

            # Calculate occupancy for each date in range
            for date in date_range:
                date_str = date.strftime(date_format)

                # Count admissions before or on this date with discharge after this date
                occupancy = ((admissions_df['admittime'] <= date) &
                             (admissions_df['dischtime'] > date)).sum()

                occupancy_data.append({
                    'Date': date,
                    'Occupancy': occupancy
                })

            # Create dataframe
            occupancy_df = pd.DataFrame(occupancy_data)

            # Plot occupancy
            fig = px.line(
                occupancy_df,
                x='Date',
                y='Occupancy',
                title=f'Bed Occupancy Over Time ({time_granularity})',
                color_discrete_sequence=['#4682b4']
            )
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Bed Occupancy'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Admission and discharge time information not available for occupancy analysis.")

    def length_of_stay(self):
        """Display length of stay distributions."""
        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for length of stay analysis.")
            return

        admissions_df = self.data['admissions']

        if 'los_days' not in admissions_df.columns:
            st.warning("Length of stay information not available in the dataset.")
            return

        st.subheader("Length of Stay Distributions")

        # Overall LOS distribution
        st.write("#### Overall Length of Stay Distribution")

        # Allow user to filter extreme values
        max_los = st.slider(
            "Maximum Length of Stay (days) to Display",
            min_value=1,
            max_value=int(admissions_df['los_days'].max()),
            value=30
        )

        # Filter data
        los_filtered = admissions_df[admissions_df['los_days'] <= max_los]

        # Plot histogram
        fig = px.histogram(
            los_filtered,
            x='los_days',
            nbins=30,
            color_discrete_sequence=['#4682b4'],
            title=f'Length of Stay Distribution (≤ {max_los} days)'
        )
        fig.update_layout(
            xaxis_title='Length of Stay (days)',
            yaxis_title='Count',
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)

        # LOS statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Median LOS (days)", f"{admissions_df['los_days'].median():.1f}")

        with col2:
            st.metric("Mean LOS (days)", f"{admissions_df['los_days'].mean():.1f}")

        with col3:
            st.metric("Min LOS (days)", f"{admissions_df['los_days'].min():.1f}")

        with col4:
            st.metric("Max LOS (days)", f"{admissions_df['los_days'].max():.1f}")

        # LOS by department/care unit
        if 'transfers' in self.data and self.data['transfers'] is not None and 'careunit' in self.data['transfers'].columns:
            st.write("#### Length of Stay by Department/Care Unit")

            transfers_df = self.data['transfers']

            # Get top 10 care units by patient count
            top_units = transfers_df['careunit'].value_counts().nlargest(10).index.tolist()

            # Filter for top units and calculate LOS
            unit_los = transfers_df[transfers_df['careunit'].isin(top_units)]

            # Create box plot
            if 'unit_los_hours' in unit_los.columns:
                # Convert hours to days for plotting
                unit_los['unit_los_days'] = unit_los['unit_los_hours'] / 24

                fig = px.box(
                    unit_los,
                    x='careunit',
                    y='unit_los_days',
                    color='careunit',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='Length of Stay by Care Unit (Top 10)',
                    points="outliers",
                    labels={'careunit': 'Care Unit', 'unit_los_days': 'Length of Stay (days)'}
                )
                fig.update_layout(
                    xaxis_title='Care Unit',
                    yaxis_title='Length of Stay (days)',
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Unit length of stay information not available in the dataset.")
        else:
            st.info("Care unit information not available in the dataset.")

        # LOS by admission type
        if 'admission_type' in admissions_df.columns:
            st.write("#### Length of Stay by Admission Type")

            fig = px.box(
                los_filtered,
                x='admission_type',
                y='los_days',
                color='admission_type',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title=f'Length of Stay by Admission Type (≤ {max_los} days)',
                points="outliers",
                labels={'admission_type': 'Admission Type', 'los_days': 'Length of Stay (days)'}
            )
            fig.update_layout(
                xaxis_title='Admission Type',
                yaxis_title='Length of Stay (days)',
                xaxis={'categoryorder':'total descending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Admission type information not available in the dataset.")

    def time_series_analysis(self):
        """Display interactive time series analysis of admission/discharge patterns."""
        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for time series analysis.")
            return

        admissions_df = self.data['admissions']

        if 'admittime' not in admissions_df.columns or 'dischtime' not in admissions_df.columns:
            st.warning("Admission and discharge time information not available for time series analysis.")
            return

        st.subheader("Interactive Time Series Analysis")

        # Time range selector
        st.write("#### Select Time Range")

        # Get min and max dates
        min_date = admissions_df['admittime'].min().date()
        max_date = admissions_df['dischtime'].max().date()

        # Date range selector
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Ensure we have start and end dates
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = min_date
            end_date = max_date

        # Convert to datetime for filtering
        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Filter data by date range
        filtered_admissions = admissions_df[
            ((admissions_df['admittime'] >= start_datetime) & (admissions_df['admittime'] <= end_datetime)) |
            ((admissions_df['dischtime'] >= start_datetime) & (admissions_df['dischtime'] <= end_datetime))
        ]

        # Time granularity selector
        time_granularity = st.selectbox(
            "Select Time Granularity",
            ["Daily", "Weekly", "Monthly"],
            index=1
        )

        # Admission/discharge patterns
        st.write("#### Admission and Discharge Patterns")

        # Create date range based on granularity
        if time_granularity == "Daily":
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_format = '%Y-%m-%d'
        elif time_granularity == "Weekly":
            date_range = pd.date_range(start=start_date, end=end_date, freq='W')
            date_format = '%Y-%m-%d'
        else:  # Monthly
            date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
            date_format = '%Y-%m'

        # Initialize data
        time_series_data = []

        # Calculate admissions and discharges for each date in range
        for date in date_range:
            date_str = date.strftime(date_format)

            if time_granularity == "Daily":
                # Daily: count events on this exact date
                admissions = (admissions_df['admittime'].dt.date == date.date()).sum()
                discharges = (admissions_df['dischtime'].dt.date == date.date()).sum()
            elif time_granularity == "Weekly":
                # Weekly: count events in this week
                week_start = date
                week_end = date + pd.Timedelta(days=6)
                admissions = ((admissions_df['admittime'] >= week_start) &
                              (admissions_df['admittime'] <= week_end)).sum()
                discharges = ((admissions_df['dischtime'] >= week_start) &
                              (admissions_df['dischtime'] <= week_end)).sum()
            else:  # Monthly
                # Monthly: count events in this month
                month_start = date
                month_end = date + pd.Timedelta(days=32)
                month_end = month_end.replace(day=1) - pd.Timedelta(days=1)
                admissions = ((admissions_df['admittime'] >= month_start) &
                              (admissions_df['admittime'] <= month_end)).sum()
                discharges = ((admissions_df['dischtime'] >= month_start) &
                              (admissions_df['dischtime'] <= month_end)).sum()

            time_series_data.append({
                'Date': date,
                'Type': 'Admissions',
                'Count': admissions
            })

            time_series_data.append({
                'Date': date,
                'Type': 'Discharges',
                'Count': discharges
            })

        # Create dataframe
        time_series_df = pd.DataFrame(time_series_data)

        # Plot time series
        fig = px.line(
            time_series_df,
            x='Date',
            y='Count',
            color='Type',
            title=f'Admission and Discharge Patterns ({time_granularity})',
            color_discrete_map={'Admissions': '#4682b4', 'Discharges': '#e41a1c'}
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Count',
            legend_title='Event Type'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Seasonal patterns
        st.write("#### Seasonal Patterns")

        # Allow user to select pattern type
        pattern_type = st.selectbox(
            "Select Pattern Type",
            ["Day of Week", "Month of Year", "Hour of Day"],
            index=0
        )

        # Initialize data
        seasonal_data = []

        if pattern_type == "Day of Week":
            # Day of week patterns
            for day in range(7):
                day_name = calendar.day_name[day]

                admissions = (admissions_df['admittime'].dt.dayofweek == day).sum()
                discharges = (admissions_df['dischtime'].dt.dayofweek == day).sum()

                seasonal_data.append({
                    'Period': day_name,
                    'Type': 'Admissions',
                    'Count': admissions
                })

                seasonal_data.append({
                    'Period': day_name,
                    'Type': 'Discharges',
                    'Count': discharges
                })

            # Create dataframe
            seasonal_df = pd.DataFrame(seasonal_data)

            # Ensure days are in correct order
            day_order = list(calendar.day_name)

            # Plot patterns
            fig = px.bar(
                seasonal_df,
                x='Period',
                y='Count',
                color='Type',
                barmode='group',
                title='Admission and Discharge Patterns by Day of Week',
                color_discrete_map={'Admissions': '#4682b4', 'Discharges': '#e41a1c'},
                category_orders={'Period': day_order}
            )
            fig.update_layout(
                xaxis_title='Day of Week',
                yaxis_title='Count',
                legend_title='Event Type'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif pattern_type == "Month of Year":
            # Month of year patterns
            for month in range(1, 13):
                month_name = calendar.month_name[month]

                admissions = (admissions_df['admittime'].dt.month == month).sum()
                discharges = (admissions_df['dischtime'].dt.month == month).sum()

                seasonal_data.append({
                    'Period': month_name,
                    'Type': 'Admissions',
                    'Count': admissions
                })

                seasonal_data.append({
                    'Period': month_name,
                    'Type': 'Discharges',
                    'Count': discharges
                })

            # Create dataframe
            seasonal_df = pd.DataFrame(seasonal_data)

            # Ensure months are in correct order
            month_order = list(calendar.month_name)[1:]

            # Plot patterns
            fig = px.bar(
                seasonal_df,
                x='Period',
                y='Count',
                color='Type',
                barmode='group',
                title='Admission and Discharge Patterns by Month of Year',
                color_discrete_map={'Admissions': '#4682b4', 'Discharges': '#e41a1c'},
                category_orders={'Period': month_order}
            )
            fig.update_layout(
                xaxis_title='Month of Year',
                yaxis_title='Count',
                legend_title='Event Type'
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Hour of Day
            # Hour of day patterns
            for hour in range(24):
                hour_label = f"{hour:02d}:00"

                admissions = (admissions_df['admittime'].dt.hour == hour).sum()
                discharges = (admissions_df['dischtime'].dt.hour == hour).sum()

                seasonal_data.append({
                    'Period': hour_label,
                    'Type': 'Admissions',
                    'Count': admissions
                })

                seasonal_data.append({
                    'Period': hour_label,
                    'Type': 'Discharges',
                    'Count': discharges
                })

            # Create dataframe
            seasonal_df = pd.DataFrame(seasonal_data)

            # Plot patterns
            fig = px.bar(
                seasonal_df,
                x='Period',
                y='Count',
                color='Type',
                barmode='group',
                title='Admission and Discharge Patterns by Hour of Day',
                color_discrete_map={'Admissions': '#4682b4', 'Discharges': '#e41a1c'}
            )
            fig.update_layout(
                xaxis_title='Hour of Day',
                yaxis_title='Count',
                legend_title='Event Type'
            )
            st.plotly_chart(fig, use_container_width=True)
