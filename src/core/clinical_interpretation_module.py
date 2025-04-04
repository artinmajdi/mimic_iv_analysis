import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import calendar
from collections import Counter
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClinicalInterpretation:
    """Class for clinical interpretation of discharge patterns in MIMIC-IV dataset."""

    def __init__(self, data_loader):
        """Initialize the clinical interpretation module with a data loader.

        Args:
            data_loader (MIMICDataLoader): Data loader object with preprocessed data
        """
        self.data_loader = data_loader
        self.data = data_loader.preprocessed

    def provider_order_patterns(self):
        """Display provider order pattern comparison."""
        if 'poe' not in self.data or self.data['poe'] is None:
            st.warning("Provider order entry data not available for pattern analysis.")
            return

        poe_df = self.data['poe']

        st.subheader("Provider Order Pattern Comparison")

        # Check if provider information is available
        if 'provider_id' not in poe_df.columns:
            st.warning("Provider ID information not available in the POE data.")
            st.info("""
            Provider order pattern comparison requires provider identification in the dataset.
            This analysis would typically include:
            1. Identification of providers and their ordering patterns
            2. Comparison of order types, timing, and sequences across providers
            3. Analysis of provider-specific discharge practices
            4. Visualization of provider variability in care patterns
            """)

            # Use example data for demonstration
            use_example_data = True
        else:
            # We have the necessary data, ask if user wants to use real data or example data
            use_example_data = not st.checkbox("Use actual MIMIC-IV data for analysis", value=False,
                                           help="If unchecked, example data will be used for demonstration")

        # Provider selection
        st.write("#### Select Providers for Comparison")

        if not use_example_data:
            # Get list of providers with sufficient orders
            provider_counts = poe_df['provider_id'].value_counts()
            active_providers = provider_counts[provider_counts >= 50].index.tolist()

            if len(active_providers) > 0:
                selected_providers = st.multiselect(
                    "Select providers to compare",
                    options=active_providers,
                    default=active_providers[:3] if len(active_providers) >= 3 else active_providers
                )
            else:
                st.warning("No providers with sufficient orders found in the dataset.")
                selected_providers = []
        else:
            # Example provider IDs
            provider_options = ["Provider A", "Provider B", "Provider C", "Provider D", "Provider E"]
            selected_providers = st.multiselect(
                "Select providers to compare",
                options=provider_options,
                default=provider_options[:3]
            )

        # Analysis type
        st.write("#### Analysis Type")
        analysis_type = st.radio(
            "Select analysis type",
            options=["Order Type Distribution", "Order Timing Patterns", "Discharge Order Sequences"],
            index=0
        )

        # Proceed with analysis if providers are selected
        if len(selected_providers) > 0:
            if use_example_data:
                # Create example data for demonstration
                st.write(f"#### {analysis_type} (Example Data)")

                if analysis_type == "Order Type Distribution":
                    # Generate example order type distribution data
                    order_types = ["Medications", "Labs", "Imaging", "Procedures", "Nursing", "Diet", "Other"]

                    data = []
                    for provider in selected_providers:
                        # Generate random distribution with some provider-specific patterns
                        if provider == "Provider A":
                            # Provider A orders more medications and labs
                            weights = [0.35, 0.25, 0.1, 0.1, 0.1, 0.05, 0.05]
                        elif provider == "Provider B":
                            # Provider B orders more imaging and procedures
                            weights = [0.2, 0.15, 0.25, 0.2, 0.1, 0.05, 0.05]
                        elif provider == "Provider C":
                            # Provider C has more balanced ordering
                            weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]
                        elif provider == "Provider D":
                            # Provider D focuses on nursing and diet orders
                            weights = [0.2, 0.15, 0.1, 0.1, 0.25, 0.15, 0.05]
                        else:
                            # Default distribution
                            weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]

                        # Create data entries
                        for i, order_type in enumerate(order_types):
                            data.append({
                                "Provider": provider,
                                "Order Type": order_type,
                                "Percentage": weights[i] * 100
                            })

                    # Create dataframe
                    df = pd.DataFrame(data)

                    # Create grouped bar chart
                    fig = px.bar(
                        df,
                        x="Order Type",
                        y="Percentage",
                        color="Provider",
                        barmode="group",
                        title="Order Type Distribution by Provider",
                        labels={"Percentage": "Percentage of Orders (%)"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The chart above shows the distribution of order types for each provider. Notable patterns include:
                    - **Provider A** tends to order more medications and laboratory tests compared to others
                    - **Provider B** has a higher proportion of imaging and procedure orders
                    - **Provider C** shows a more balanced distribution across order types
                    - **Provider D** places more emphasis on nursing and dietary orders

                    These patterns may reflect different specialties, practice styles, or patient populations.
                    """)

                elif analysis_type == "Order Timing Patterns":
                    # Generate example order timing data
                    hours = list(range(24))

                    data = []
                    for provider in selected_providers:
                        # Generate random distribution with some provider-specific patterns
                        if provider == "Provider A":
                            # Provider A orders more in morning
                            base = [5, 8, 12, 20, 25, 30, 35, 40, 30, 25, 20, 15, 10, 15, 20, 25, 20, 15, 10, 8, 5, 3, 2, 3]
                        elif provider == "Provider B":
                            # Provider B orders more in afternoon
                            base = [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 40, 35, 30, 25, 20, 15, 10, 8, 5, 3, 2, 2, 2]
                        elif provider == "Provider C":
                            # Provider C orders throughout the day
                            base = [10, 15, 20, 25, 30, 35, 30, 25, 20, 25, 30, 35, 30, 25, 20, 25, 30, 25, 20, 15, 10, 5, 5, 5]
                        else:
                            # Default distribution
                            base = [5, 10, 15, 20, 25, 30, 35, 30, 25, 20, 25, 30, 25, 20, 15, 20, 25, 20, 15, 10, 5, 3, 2, 3]

                        # Add some random variation
                        np.random.seed(42 + selected_providers.index(provider))
                        counts = [max(0, b + np.random.randint(-5, 6)) for b in base]

                        # Create data entries
                        for hour, count in zip(hours, counts):
                            data.append({
                                "Provider": provider,
                                "Hour": hour,
                                "Order Count": count
                            })

                    # Create dataframe
                    df = pd.DataFrame(data)

                    # Create line chart
                    fig = px.line(
                        df,
                        x="Hour",
                        y="Order Count",
                        color="Provider",
                        title="Order Timing Patterns by Hour of Day",
                        labels={"Hour": "Hour of Day (24h)", "Order Count": "Number of Orders"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    fig.update_layout(
                        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The chart above shows the timing of orders throughout the day for each provider. Notable patterns include:
                    - **Provider A** tends to place most orders in the morning (6-10 AM)
                    - **Provider B** has peak ordering activity in the late morning to early afternoon (10 AM-2 PM)
                    - **Provider C** shows a more consistent ordering pattern throughout the day

                    These timing differences may reflect rounding schedules, shift patterns, or workflow preferences.
                    Alignment of order timing with hospital workflows can impact discharge efficiency.
                    """)

                elif analysis_type == "Discharge Order Sequences":
                    # Generate example discharge order sequence data
                    sequence_data = {
                        "Provider A": [
                            "Labs → Medication Adjustment → Discharge Planning → Discharge Order",
                            "Imaging → Consult → Medication Adjustment → Discharge Order",
                            "Medication Adjustment → Discharge Planning → Discharge Order"
                        ],
                        "Provider B": [
                            "Consult → Discharge Planning → Medication Adjustment → Discharge Order",
                            "Imaging → Labs → Discharge Planning → Discharge Order",
                            "Procedure → Medication Adjustment → Discharge Planning → Discharge Order"
                        ],
                        "Provider C": [
                            "Discharge Planning → Labs → Medication Adjustment → Discharge Order",
                            "Discharge Planning → Consult → Discharge Order",
                            "Medication Adjustment → Discharge Planning → Discharge Order"
                        ]
                    }

                    # Filter to selected providers
                    sequence_data = {p: sequence_data[p] for p in selected_providers if p in sequence_data}

                    # Display sequences for each provider
                    for provider, sequences in sequence_data.items():
                        st.write(f"**{provider}'s Common Discharge Sequences:**")
                        for i, seq in enumerate(sequences, 1):
                            st.write(f"{i}. {seq}")
                        st.write("")

                    # Create network visualization of order sequences
                    st.write("**Order Sequence Network Visualization:**")

                    # Create a simple network graph for visualization
                    G = nx.DiGraph()

                    # Add nodes for common order types
                    nodes = ["Labs", "Imaging", "Consult", "Procedure", "Medication Adjustment",
                             "Discharge Planning", "Discharge Order"]
                    for node in nodes:
                        G.add_node(node)

                    # Add edges based on sequences
                    edges = [
                        ("Labs", "Medication Adjustment"),
                        ("Medication Adjustment", "Discharge Planning"),
                        ("Discharge Planning", "Discharge Order"),
                        ("Imaging", "Consult"),
                        ("Consult", "Medication Adjustment"),
                        ("Consult", "Discharge Planning"),
                        ("Imaging", "Labs"),
                        ("Procedure", "Medication Adjustment"),
                        ("Discharge Planning", "Labs")
                    ]

                    for edge in edges:
                        G.add_edge(edge[0], edge[1])

                    # Create positions for nodes
                    pos = nx.spring_layout(G, seed=42)

                    # Create edge traces
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines')

                    # Create node traces
                    node_x = []
                    node_y = []
                    node_text = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        hoverinfo='text',
                        marker=dict(
                            showscale=False,
                            color='#6175c1',
                            size=20,
                            line_width=2))

                    # Create figure
                    fig = go.Figure(data=[edge_trace, node_trace],
                                 layout=go.Layout(
                                    title="Common Order Sequence Network",
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The network visualization and sequence lists show common order patterns leading to discharge:

                    - **Provider A** typically follows a lab-first approach, checking results before medication adjustments
                    - **Provider B** often starts with consultations or imaging before proceeding to discharge planning
                    - **Provider C** tends to initiate discharge planning earlier in the sequence

                    Earlier initiation of discharge planning (as seen with Provider C) may be associated with more efficient discharges.
                    """)
            else:
                # Use actual MIMIC-IV data
                st.info(f"Analysis of {analysis_type} using actual MIMIC-IV data would be implemented here.")
                st.warning("This feature is not implemented in the current version.")
        else:
            st.warning("Please select at least one provider for comparison.")

    def unit_specific_analysis(self):
        """Display unit-specific discharge pattern analysis."""
        if 'transfers' not in self.data or self.data['transfers'] is None:
            st.warning("Transfers data not available for unit-specific analysis.")
            return

        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for unit-specific analysis.")
            return

        transfers_df = self.data['transfers']
        admissions_df = self.data['admissions']

        st.subheader("Unit-Specific Discharge Pattern Analysis")

        # Check if care unit information is available
        if 'careunit' not in transfers_df.columns:
            st.warning("Care unit information not available in the transfers data.")
            st.info("""
            Unit-specific discharge pattern analysis requires care unit information in the dataset.
            This analysis would typically include:
            1. Comparison of discharge patterns across different hospital units
            2. Analysis of length of stay and discharge timing by unit
            3. Identification of unit-specific bottlenecks in the discharge process
            4. Visualization of patient flow patterns between units
            """)

            # Use example data for demonstration
            use_example_data = True
        else:
            # We have the necessary data, ask if user wants to use real data or example data
            use_example_data = not st.checkbox("Use actual MIMIC-IV data for analysis", value=False,
                                           help="If unchecked, example data will be used for demonstration")

        # Unit selection
        st.write("#### Select Units for Analysis")

        if not use_example_data:
            # Get list of care units
            care_units = sorted(transfers_df['careunit'].unique())

            if len(care_units) > 0:
                selected_units = st.multiselect(
                    "Select care units to analyze",
                    options=care_units,
                    default=care_units[:3] if len(care_units) >= 3 else care_units
                )
            else:
                st.warning("No care units found in the dataset.")
                selected_units = []
        else:
            # Example care units
            unit_options = ["Medical ICU", "Surgical ICU", "Medical Ward", "Surgical Ward", "Step-Down Unit"]
            selected_units = st.multiselect(
                "Select care units to analyze",
                options=unit_options,
                default=unit_options[:3]
            )

        # Analysis type
        st.write("#### Analysis Type")
        analysis_type = st.radio(
            "Select analysis type",
            options=["Discharge Timing", "Length of Stay", "Discharge Destination", "Readmission Rates"],
            index=0
        )

        # Proceed with analysis if units are selected
        if len(selected_units) > 0:
            if use_example_data:
                # Create example data for demonstration
                st.write(f"#### {analysis_type} by Unit (Example Data)")

                if analysis_type == "Discharge Timing":
                    # Generate example discharge timing data
                    hours = list(range(24))

                    data = []
                    for unit in selected_units:
                        # Generate random distribution with some unit-specific patterns
                        if unit == "Medical ICU":
                            # Medical ICU discharges more in morning
                            base = [1, 1, 1, 2, 3, 5, 8, 12, 15, 18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1]
                        elif unit == "Surgical ICU":
                            # Surgical ICU discharges more in afternoon
                            base = [1, 1, 1, 1, 2, 3, 5, 8, 10, 12, 15, 18, 15, 12, 10, 8, 6, 5, 3, 2, 1, 1, 1, 1]
                        elif unit == "Medical Ward":
                            # Medical Ward discharges throughout the day
                            base = [1, 1, 1, 2, 3, 5, 8, 10, 12, 15, 18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1, 1, 1, 1]
                        elif unit == "Surgical Ward":
                            # Surgical Ward discharges more in late morning
                            base = [1, 1, 1, 1, 2, 3, 5, 8, 12, 15, 18, 15, 12, 10, 8, 6, 5, 3, 2, 1, 1, 1, 1, 1]
                        else:
                            # Default distribution
                            base = [1, 1, 1, 2, 3, 5, 8, 10, 12, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1]

                        # Add some random variation
                        np.random.seed(42 + selected_units.index(unit))
                        counts = [max(0, b + np.random.randint(-2, 3)) for b in base]

                        # Create data entries
                        for hour, count in zip(hours, counts):
                            data.append({
                                "Unit": unit,
                                "Hour": hour,
                                "Discharge Count": count
                            })

                    # Create dataframe
                    df = pd.DataFrame(data)

                    # Create line chart
                    fig = px.line(
                        df,
                        x="Hour",
                        y="Discharge Count",
                        color="Unit",
                        title="Discharge Timing by Hour of Day",
                        labels={"Hour": "Hour of Day (24h)", "Discharge Count": "Number of Discharges"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    fig.update_layout(
                        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The chart above shows discharge timing patterns throughout the day for each unit:

                    - **Medical ICU** tends to discharge patients in the morning (8-11 AM)
                    - **Surgical ICU** shows peak discharge activity in the early afternoon (11 AM-2 PM)
                    - **Medical Ward** has a broader discharge window spanning morning to afternoon

                    These timing differences may reflect unit workflows, physician rounding schedules, or bed management policies.
                    Early morning discharges can improve hospital throughput by making beds available for new admissions.
                    """)

                elif analysis_type == "Length of Stay":
                    # Generate example length of stay data
                    data = []

                    for unit in selected_units:
                        # Generate random LOS with unit-specific patterns
                        if unit == "Medical ICU":
                            # Medical ICU: shorter stays
                            mean_los = 3.5
                            std_los = 2.0
                        elif unit == "Surgical ICU":
                            # Surgical ICU: medium stays
                            mean_los = 4.2
                            std_los = 2.5
                        elif unit == "Medical Ward":
                            # Medical Ward: longer stays
                            mean_los = 5.8
                            std_los = 3.0
                        elif unit == "Surgical Ward":
                            # Surgical Ward: medium stays
                            mean_los = 4.5
                            std_los = 2.2
                        elif unit == "Step-Down Unit":
                            # Step-Down Unit: variable stays
                            mean_los = 3.8
                            std_los = 2.8
                        else:
                            # Default
                            mean_los = 4.0
                            std_los = 2.0

                        # Generate 100 samples for each unit
                        np.random.seed(42 + selected_units.index(unit))
                        los_values = np.random.lognormal(mean=np.log(mean_los), sigma=0.5, size=100)

                        # Create data entries
                        for los in los_values:
                            data.append({
                                "Unit": unit,
                                "Length of Stay (days)": los
                            })

                    # Create dataframe
                    df = pd.DataFrame(data)

                    # Create box plot
                    fig = px.box(
                        df,
                        x="Unit",
                        y="Length of Stay (days)",
                        color="Unit",
                        title="Length of Stay Distribution by Unit",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate summary statistics
                    summary = df.groupby("Unit")["Length of Stay (days)"].agg(["mean", "median", "std", "min", "max"]).reset_index()
                    summary = summary.round(2)

                    # Display summary table
                    st.write("**Length of Stay Summary Statistics (days):**")
                    st.dataframe(summary)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The box plot and summary statistics show length of stay patterns by unit:

                    - **Medical ICU** has shorter stays, typical for critical care units where patients are either stabilized quickly or transferred
                    - **Medical Ward** shows longer and more variable stays, reflecting the complex medical conditions treated there
                    - **Surgical units** show moderate length of stay with less variability, suggesting more predictable recovery trajectories

                    Understanding these patterns can help identify opportunities for reducing unnecessary hospital days and improving patient flow.
                    """)

                elif analysis_type == "Discharge Destination":
                    # Generate example discharge destination data
                    destinations = ["Home", "Skilled Nursing Facility", "Rehabilitation", "Another Hospital", "Expired"]

                    data = []
                    for unit in selected_units:
                        # Generate random distribution with some unit-specific patterns
                        if unit == "Medical ICU":
                            # Medical ICU: more transfers to other units, higher mortality
                            weights = [0.4, 0.15, 0.1, 0.25, 0.1]
                        elif unit == "Surgical ICU":
                            # Surgical ICU: more transfers to step-down, moderate mortality
                            weights = [0.35, 0.2, 0.15, 0.22, 0.08]
                        elif unit == "Medical Ward":
                            # Medical Ward: more to home and SNF
                            weights = [0.6, 0.25, 0.05, 0.07, 0.03]
                        elif unit == "Surgical Ward":
                            # Surgical Ward: more to home and rehab
                            weights = [0.65, 0.1, 0.2, 0.03, 0.02]
                        elif unit == "Step-Down Unit":
                            # Step-Down Unit: mostly to home
                            weights = [0.7, 0.15, 0.1, 0.03, 0.02]
                        else:
                            # Default
                            weights = [0.6, 0.2, 0.1, 0.05, 0.05]

                        # Create data entries
                        for i, dest in enumerate(destinations):
                            data.append({
                                "Unit": unit,
                                "Discharge Destination": dest,
                                "Percentage": weights[i] * 100
                            })

                    # Create dataframe
                    df = pd.DataFrame(data)

                    # Create grouped bar chart
                    fig = px.bar(
                        df,
                        x="Discharge Destination",
                        y="Percentage",
                        color="Unit",
                        barmode="group",
                        title="Discharge Destination by Unit",
                        labels={"Percentage": "Percentage of Discharges (%)"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The chart shows discharge destination patterns by unit:

                    - **ICU units** have higher rates of transfers to other hospital units and higher mortality rates
                    - **Medical Ward** patients are more frequently discharged to skilled nursing facilities
                    - **Surgical Ward** patients have higher rates of discharge to rehabilitation facilities
                    - **Step-Down Unit** has the highest rate of discharge to home

                    These patterns reflect the different patient populations and clinical needs in each unit.
                    Understanding discharge destination patterns can help in resource planning and care coordination.
                    """)

                elif analysis_type == "Readmission Rates":
                    # Generate example readmission data
                    time_windows = ["7 Days", "14 Days", "30 Days", "90 Days"]

                    data = []
                    for unit in selected_units:
                        # Generate random readmission rates with unit-specific patterns
                        if unit == "Medical ICU":
                            # Medical ICU: higher readmission rates
                            base_rates = [8, 12, 18, 24]
                        elif unit == "Surgical ICU":
                            # Surgical ICU: moderate readmission rates
                            base_rates = [6, 10, 15, 22]
                        elif unit == "Medical Ward":
                            # Medical Ward: higher readmission rates
                            base_rates = [9, 14, 20, 26]
                        elif unit == "Surgical Ward":
                            # Surgical Ward: lower readmission rates
                            base_rates = [5, 8, 14, 20]
                        elif unit == "Step-Down Unit":
                            # Step-Down Unit: moderate readmission rates
                            base_rates = [7, 11, 16, 23]
                        else:
                            # Default
                            base_rates = [7, 12, 18, 24]

                        # Add some random variation
                        np.random.seed(42 + selected_units.index(unit))
                        rates = [max(0, r + np.random.randint(-2, 3)) for r in base_rates]

                        # Create data entries
                        for i, window in enumerate(time_windows):
                            data.append({
                                "Unit": unit,
                                "Time Window": window,
                                "Readmission Rate": rates[i]
                            })

                    # Create dataframe
                    df = pd.DataFrame(data)

                    # Create grouped bar chart
                    fig = px.bar(
                        df,
                        x="Time Window",
                        y="Readmission Rate",
                        color="Unit",
                        barmode="group",
                        title="Readmission Rates by Unit and Time Window",
                        labels={"Readmission Rate": "Readmission Rate (%)"},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Add risk factor analysis
                    st.write("#### Readmission Risk Factor Analysis")

                    # Create example risk factor data
                    risk_factors = [
                        "Multiple Comorbidities",
                        "Polypharmacy (>8 meds)",
                        "Prior Admission (90 days)",
                        "Complex Wound Care",
                        "Limited Social Support"
                    ]

                    risk_data = []
                    for unit in selected_units:
                        # Generate unit-specific risk factor odds ratios
                        if unit == "Medical ICU":
                            odds_ratios = [2.8, 1.9, 2.3, 1.5, 1.7]
                        elif unit == "Medical Ward":
                            odds_ratios = [2.5, 2.2, 2.1, 1.4, 2.0]
                        elif unit == "Surgical Ward":
                            odds_ratios = [1.8, 1.6, 1.9, 2.2, 1.5]
                        else:
                            # Default values for other units
                            odds_ratios = [2.0, 1.8, 2.0, 1.7, 1.8]

                        # Add some random variation
                        np.random.seed(100 + selected_units.index(unit))
                        odds_ratios = [max(1.0, or_val + np.random.uniform(-0.3, 0.3)) for or_val in odds_ratios]

                        # Create data entries
                        for i, factor in enumerate(risk_factors):
                            risk_data.append({
                                "Unit": unit,
                                "Risk Factor": factor,
                                "Odds Ratio": round(odds_ratios[i], 2)
                            })

                    # Create dataframe
                    risk_df = pd.DataFrame(risk_data)

                    # Create heatmap
                    risk_pivot = risk_df.pivot(index="Risk Factor", columns="Unit", values="Odds Ratio")

                    fig = px.imshow(
                        risk_pivot,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Readmission Risk Factors by Unit (Odds Ratios)",
                        labels={"color": "Odds Ratio"}
                    )

                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Add interpretation
                    st.write("**Interpretation:**")
                    st.write("""
                    The charts above show readmission patterns by unit:

                    - **Readmission rates** increase with longer time windows across all units
                    - **Medical units** (both ward and ICU) show higher readmission rates than surgical units
                    - **Risk factor analysis** reveals that:
                      - Multiple comorbidities are the strongest predictor of readmission for ICU patients
                      - Polypharmacy has a particularly strong effect in medical ward patients
                      - Complex wound care is a significant risk factor for surgical patients
                      - Limited social support affects readmission across all units

                    These patterns can help target discharge interventions to patients at highest risk of readmission.
                    For example, enhanced medication reconciliation for medical ward patients or improved wound care
                    education for surgical patients could help reduce readmissions.
                    """)
            else:
                # Use actual MIMIC-IV data
                st.info(f"Analysis of {analysis_type} using actual MIMIC-IV data would be implemented here.")
                st.warning("This feature is not implemented in the current version.")
        else:
            st.warning("Please select at least one unit for analysis.")

    def discharge_order_analysis(self):
        """Display discharge order pattern analysis."""
        if 'poe' not in self.data or self.data['poe'] is None:
            st.warning("Provider order entry data not available for discharge order analysis.")
            return

        if 'admissions' not in self.data or self.data['admissions'] is None:
            st.warning("Admissions data not available for discharge order analysis.")
            return

        poe_df = self.data['poe']
        admissions_df = self.data['admissions']

        st.subheader("Discharge Order Pattern Analysis")

        # Check if order information is available
        if 'order_type' not in poe_df.columns or 'order_subtype' not in poe_df.columns:
            st.warning("Order type information not available in the POE data.")
            st.info("""
            Discharge order pattern analysis requires order type information in the dataset.
            This analysis would typically include:
            1. Identification of common discharge order sequences
            2. Analysis of timing between discharge orders
            3. Comparison of discharge order patterns across different units and providers
            4. Identification of discharge order patterns associated with readmissions
            """)

            # Use example data for demonstration
            use_example_data = True
        else:
            # We have the necessary data, ask if user wants to use real data or example data
            use_example_data = not st.checkbox("Use actual MIMIC-IV data for analysis", value=False,
                                           help="If unchecked, example data will be used for demonstration")

        # Analysis type
        st.write("#### Analysis Type")
        analysis_type = st.radio(
            "Select analysis type",
            options=["Discharge Order Sequence", "Time to Discharge", "Discharge Order Compliance"],
            index=0
        )

        if use_example_data:
            # Create example data for demonstration
            st.write(f"#### {analysis_type} (Example Data)")

            if analysis_type == "Discharge Order Sequence":
                # Generate example discharge order sequence data
                order_sequences = [
                    "Discharge Order → Medication Reconciliation → Patient Education → Follow-up Appointment",
                    "Discharge Order → Follow-up Appointment → Medication Reconciliation → Patient Education",
                    "Medication Reconciliation → Discharge Order → Patient Education → Follow-up Appointment",
                    "Medication Reconciliation → Discharge Order → Follow-up Appointment → Patient Education",
                    "Patient Education → Medication Reconciliation → Discharge Order → Follow-up Appointment"
                ]

                sequence_counts = [42, 35, 28, 22, 15]
                total_discharges = sum(sequence_counts)

                # Create dataframe
                df = pd.DataFrame({
                    "Order Sequence": order_sequences,
                    "Count": sequence_counts,
                    "Percentage": [count/total_discharges*100 for count in sequence_counts]
                })

                # Create bar chart
                fig = px.bar(
                    df,
                    y="Order Sequence",
                    x="Percentage",
                    orientation='h',
                    title="Common Discharge Order Sequences",
                    labels={"Percentage": "Percentage of Discharges (%)"},
                    color="Percentage",
                    color_continuous_scale="Viridis",
                    text=df["Count"].apply(lambda x: f"{x} discharges")
                )

                fig.update_layout(
                    height=400,
                    yaxis={'categoryorder':'total ascending'}
                )
                fig.update_traces(textposition='outside')

                st.plotly_chart(fig, use_container_width=True)

                # Add sequence network visualization
                st.write("**Discharge Order Sequence Network:**")

                # Create a simple network graph for visualization
                G = nx.DiGraph()

                # Add nodes for common order types
                nodes = ["Discharge Order", "Medication Reconciliation", "Patient Education", "Follow-up Appointment"]
                for node in nodes:
                    G.add_node(node)

                # Add edges based on sequences with weights
                edges = [
                    ("Discharge Order", "Medication Reconciliation", 42),
                    ("Medication Reconciliation", "Patient Education", 42),
                    ("Patient Education", "Follow-up Appointment", 42),
                    ("Discharge Order", "Follow-up Appointment", 35),
                    ("Follow-up Appointment", "Medication Reconciliation", 35),
                    ("Medication Reconciliation", "Patient Education", 35),
                    ("Medication Reconciliation", "Discharge Order", 28),
                    ("Discharge Order", "Patient Education", 28),
                    ("Patient Education", "Follow-up Appointment", 28),
                    ("Medication Reconciliation", "Discharge Order", 22),
                    ("Discharge Order", "Follow-up Appointment", 22),
                    ("Follow-up Appointment", "Patient Education", 22),
                    ("Patient Education", "Medication Reconciliation", 15),
                    ("Medication Reconciliation", "Discharge Order", 15),
                    ("Discharge Order", "Follow-up Appointment", 15)
                ]

                for edge in edges:
                    G.add_edge(edge[0], edge[1], weight=edge[2])

                # Create positions for nodes
                pos = nx.spring_layout(G, seed=42)

                # Create edge traces with varying widths based on weight
                edge_x = []
                edge_y = []
                edge_traces = []

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    weight = edge[2]['weight']

                    trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=weight/10, color='#888'),
                        hoverinfo='text',
                        text=f"{edge[0]} → {edge[1]} ({weight} discharges)",
                        mode='lines'
                    )
                    edge_traces.append(trace)

                # Create node trace
                node_x = []
                node_y = []
                node_text = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text',
                    marker=dict(
                        showscale=False,
                        color='#6175c1',
                        size=20,
                        line_width=2
                    )
                )

                # Create figure
                fig = go.Figure(data=edge_traces + [node_trace],
                             layout=go.Layout(
                                title="Discharge Order Sequence Network",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=500
                             ))

                st.plotly_chart(fig, use_container_width=True)

                # Add interpretation
                st.write("**Interpretation:**")
                st.write("""
                The analysis of discharge order sequences reveals several patterns:

                - The most common sequence begins with the discharge order, followed by medication reconciliation,
                  patient education, and scheduling follow-up appointments
                - In about 28% of cases, medication reconciliation precedes the formal discharge order
                - Less commonly, patient education is completed early in the discharge process

                These patterns suggest opportunities for process standardization. Starting medication reconciliation
                earlier in the discharge process could potentially reduce delays, as it's a common prerequisite
                for the discharge order in many cases.
                """)

            elif analysis_type == "Time to Discharge":
                # Generate example time to discharge data
                order_types = [
                    "Discharge Order",
                    "Medication Reconciliation",
                    "Patient Education",
                    "Follow-up Appointment",
                    "Discharge Summary"
                ]

                # Hours before actual discharge
                mean_times = [4.2, 8.5, 6.3, 12.1, 2.8]
                std_times = [1.5, 2.8, 2.2, 4.5, 1.2]

                # Generate sample data
                np.random.seed(42)
                data = []

                for i, order_type in enumerate(order_types):
                    # Generate 50 samples for each order type
                    times = np.random.normal(mean_times[i], std_times[i], 50)
                    times = np.clip(times, 0, 48)  # Clip to reasonable range

                    for time in times:
                        data.append({
                            "Order Type": order_type,
                            "Hours Before Discharge": time
                        })

                # Create dataframe
                df = pd.DataFrame(data)

                # Create box plot
                fig = px.box(
                    df,
                    x="Order Type",
                    y="Hours Before Discharge",
                    color="Order Type",
                    title="Time from Order to Actual Discharge",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Calculate summary statistics
                summary = df.groupby("Order Type")["Hours Before Discharge"].agg(["mean", "median", "std"]).reset_index()
                summary = summary.round(1)

                # Display summary table
                st.write("**Time to Discharge Summary Statistics (hours):**")
                st.dataframe(summary)

                # Add interpretation
                st.write("**Interpretation:**")
                st.write("""
                The analysis of time between orders and actual discharge reveals:

                - **Follow-up Appointments** are typically scheduled earliest (median 12.1 hours before discharge)
                - **Medication Reconciliation** occurs next (median 8.5 hours before discharge)
                - **Patient Education** follows (median 6.3 hours before discharge)
                - **Discharge Orders** are typically written 4.2 hours before actual discharge
                - **Discharge Summaries** are completed closest to the actual discharge time (median 2.8 hours)

                The wide variability in timing (shown by the box plot whiskers) suggests opportunities for process
                standardization. Particularly, the discharge order to actual discharge time varies significantly,
                which may indicate discharge execution inefficiencies.
                """)

            elif analysis_type == "Discharge Order Compliance":
                # Generate example compliance data
                compliance_metrics = [
                    "Complete Medication Reconciliation",
                    "Discharge Summary Completed",
                    "Follow-up Appointment Scheduled",
                    "Patient Education Documented",
                    "Discharge Instructions Provided"
                ]

                # Compliance rates by service
                services = ["Medicine", "Surgery", "Cardiology", "Neurology", "Oncology"]

                data = []
                np.random.seed(42)

                for service in services:
                    # Generate service-specific compliance rates
                    if service == "Medicine":
                        base_rates = [92, 85, 78, 88, 95]
                    elif service == "Surgery":
                        base_rates = [88, 82, 75, 80, 90]
                    elif service == "Cardiology":
                        base_rates = [95, 90, 85, 92, 98]
                    elif service == "Neurology":
                        base_rates = [90, 88, 80, 85, 93]
                    elif service == "Oncology":
                        base_rates = [93, 87, 82, 90, 96]

                    # Add some random variation
                    rates = [min(100, max(50, r + np.random.randint(-5, 6))) for r in base_rates]

                    # Create data entries
                    for i, metric in enumerate(compliance_metrics):
                        data.append({
                            "Service": service,
                            "Compliance Metric": metric,
                            "Compliance Rate": rates[i]
                        })

                # Create dataframe
                df = pd.DataFrame(data)

                # Create heatmap
                pivot_df = df.pivot(index="Compliance Metric", columns="Service", values="Compliance Rate")

                fig = px.imshow(
                    pivot_df,
                    text_auto=True,
                    color_continuous_scale="RdYlGn",
                    title="Discharge Order Compliance by Service (%)",
                    labels={"color": "Compliance Rate (%)"}
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Add interpretation
                st.write("**Interpretation:**")
                st.write("""
                The discharge order compliance analysis reveals several patterns:

                - **Cardiology** has the highest overall compliance with discharge best practices
                - **Surgery** shows the lowest compliance rates across most metrics
                - **Follow-up Appointment Scheduling** has the lowest compliance across all services
                - **Discharge Instructions** show the highest compliance rates

                These patterns suggest targeted improvement opportunities:
                1. Implement a standardized follow-up appointment scheduling process, particularly for surgical patients
                2. Share best practices from Cardiology with other services
                3. Focus on improving medication reconciliation compliance in Surgery and Neurology services
                """)
        else:
            # Use actual MIMIC-IV data
            st.info(f"Analysis of {analysis_type} using actual MIMIC-IV data would be implemented here.")
            st.warning("This feature is not implemented in the current version.")
