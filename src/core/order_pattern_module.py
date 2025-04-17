import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
import networkx as nx


class OrderPatternAnalysis:
    """Class for analyzing order patterns in MIMIC-IV dataset."""

    def __init__(self, data_loader):
        """Initialize the order pattern analysis module with a data loader.

        Args:
            data_loader (MIMICDataLoader): Data loader object with preprocessed data
        """
        self.data_loader = data_loader
        self.data = data_loader.preprocessed

    def temporal_heatmaps(self):
        """Display temporal heatmaps of order types throughout hospital stays."""
        if 'poe' not in self.data or self.data['poe'] is None:
            st.warning("Provider order entry data not available for temporal heatmap analysis.")
            return

        poe_df = self.data['poe']

        if 'order_type' not in poe_df.columns or 'ordertime' not in poe_df.columns:
            st.warning("Order type or order time information not available for temporal heatmap analysis.")
            return

        st.subheader("Temporal Heatmaps of Order Types")

        # Order type selector
        st.write("#### Select Order Types")

        # Get unique order types
        order_types = sorted(poe_df['order_type'].unique())

        # Allow user to select order types
        selected_order_types = st.multiselect(
            "Select Order Types to Analyze",
            options=order_types,
            default=order_types[:5] if len(order_types) >= 5 else order_types
        )

        if not selected_order_types:
            st.warning("Please select at least one order type.")
            return

        # Filter data for selected order types
        filtered_orders = poe_df[poe_df['order_type'].isin(selected_order_types)]

        # Time granularity selector
        time_granularity = st.selectbox(
            "Select Time Granularity",
            ["Hourly", "Daily", "Weekly"],
            index=0
        )

        # Normalize option
        normalize = st.checkbox("Normalize by Total Orders", value=True)

        # Prepare data for heatmap
        if time_granularity == "Hourly":
            # Extract hour from ordertime
            filtered_orders['time_bin'] = filtered_orders['ordertime'].dt.hour
            time_label = 'Hour of Day'
        elif time_granularity == "Daily":
            # Extract day of week from ordertime
            filtered_orders['time_bin'] = filtered_orders['ordertime'].dt.dayofweek
            filtered_orders['time_bin'] = filtered_orders['time_bin'].map(lambda x: calendar.day_name[x])
            time_label = 'Day of Week'
        else:  # Weekly
            # Extract week of year from ordertime
            filtered_orders['time_bin'] = filtered_orders['ordertime'].dt.isocalendar().week
            time_label = 'Week of Year'

        # Create pivot table for heatmap
        if time_granularity == "Hourly":
            # For hourly, we want to see distribution across days of week as well
            filtered_orders['day_of_week'] = filtered_orders['ordertime'].dt.dayofweek
            filtered_orders['day_of_week'] = filtered_orders['day_of_week'].map(lambda x: calendar.day_name[x])

            pivot_df = pd.pivot_table(
                filtered_orders,
                values='poe_id',
                index='day_of_week',
                columns=['order_type', 'time_bin'],
                aggfunc='count',
                fill_value=0
            )

            # Ensure days are in correct order
            day_order = list(calendar.day_name)
            pivot_df = pivot_df.reindex(day_order)

            # Normalize if requested
            if normalize:
                # Normalize by total orders for each day
                row_sums = pivot_df.sum(axis=1)
                pivot_df = pivot_df.div(row_sums, axis=0) * 100

            # Create heatmap for each order type
            for order_type in selected_order_types:
                if (order_type,) in pivot_df.columns.levels[0] or order_type in pivot_df.columns.levels[0]:
                    st.write(f"#### Temporal Heatmap for {order_type}")

                    # Extract data for this order type
                    try:
                        order_data = pivot_df[order_type].copy()
                    except:
                        # Handle multi-level column access
                        order_data = pivot_df.xs(order_type, axis=1, level=0)

                    # Sort columns (hours) numerically
                    order_data = order_data.reindex(sorted(order_data.columns), axis=1)

                    # Create heatmap
                    fig = px.imshow(
                        order_data,
                        labels=dict(x="Hour of Day", y="Day of Week", color="Percentage" if normalize else "Count"),
                        x=[f"{h:02d}:00" for h in range(24)] if time_granularity == "Hourly" else order_data.columns,
                        y=order_data.index,
                        color_continuous_scale="Blues",
                        title=f"Temporal Distribution of {order_type} Orders"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {order_type} with the current filters.")
        else:
            # For daily and weekly, create a simpler heatmap
            pivot_df = pd.pivot_table(
                filtered_orders,
                values='poe_id',
                index='order_type',
                columns='time_bin',
                aggfunc='count',
                fill_value=0
            )

            # Normalize if requested
            if normalize:
                # Normalize by total orders for each order type
                row_sums = pivot_df.sum(axis=1)
                pivot_df = pivot_df.div(row_sums, axis=0) * 100

            # Sort columns if they are numeric
            if time_granularity == "Weekly":
                pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
            elif time_granularity == "Daily":
                # Ensure days are in correct order
                day_order = list(calendar.day_name)
                pivot_df = pivot_df.reindex(columns=day_order)

            # Create heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x=time_label, y="Order Type", color="Percentage" if normalize else "Count"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Blues",
                title=f"Temporal Distribution of Orders by {time_label}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    def order_density(self):
        """Display order density visualization by time of day/day of week."""
        if 'poe' not in self.data or self.data['poe'] is None:
            st.warning("Provider order entry data not available for order density analysis.")
            return

        poe_df = self.data['poe']

        if 'ordertime' not in poe_df.columns:
            st.warning("Order time information not available for order density analysis.")
            return

        st.subheader("Order Density Visualization")

        # Time dimension selector
        time_dimension = st.selectbox(
            "Select Time Dimension",
            ["Time of Day", "Day of Week", "Month of Year"],
            index=0
        )

        # Order type filter
        order_types = sorted(poe_df['order_type'].unique())
        selected_order_types = st.multiselect(
            "Filter by Order Types (optional)",
            options=order_types,
            default=None
        )

        # Filter data if order types selected
        if selected_order_types:
            filtered_orders = poe_df[poe_df['order_type'].isin(selected_order_types)]
        else:
            filtered_orders = poe_df

        # Prepare data based on selected time dimension
        if time_dimension == "Time of Day":
            # Extract hour from ordertime
            filtered_orders['time_bin'] = filtered_orders['ordertime'].dt.hour
            time_label = 'Hour of Day'
            x_values = list(range(24))
            x_labels = [f"{h:02d}:00" for h in x_values]
        elif time_dimension == "Day of Week":
            # Extract day of week from ordertime
            filtered_orders['time_bin'] = filtered_orders['ordertime'].dt.dayofweek
            time_label = 'Day of Week'
            x_values = list(range(7))
            x_labels = [calendar.day_name[d] for d in x_values]
        else:  # Month of Year
            # Extract month from ordertime
            filtered_orders['time_bin'] = filtered_orders['ordertime'].dt.month
            time_label = 'Month of Year'
            x_values = list(range(1, 13))
            x_labels = [calendar.month_name[m] for m in x_values]

        # Count orders by time bin
        order_counts = filtered_orders.groupby('time_bin').size().reindex(x_values, fill_value=0)

        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            time_label: x_labels,
            'Order Count': order_counts.values
        })

        # Plot order density
        st.write(f"#### Order Density by {time_dimension}")

        fig = px.bar(
            plot_df,
            x=time_label,
            y='Order Count',
            color_discrete_sequence=['#4682b4'],
            title=f"Order Density by {time_dimension}"
        )
        fig.update_layout(
            xaxis_title=time_label,
            yaxis_title='Order Count'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Additional analysis: Order density heatmap by hour and day
        if time_dimension == "Time of Day":
            st.write("#### Order Density Heatmap by Hour and Day")

            # Extract hour and day of week
            filtered_orders['hour'] = filtered_orders['ordertime'].dt.hour
            filtered_orders['day_of_week'] = filtered_orders['ordertime'].dt.dayofweek

            # Create pivot table
            pivot_df = pd.pivot_table(
                filtered_orders,
                values='poe_id',
                index='day_of_week',
                columns='hour',
                aggfunc='count',
                fill_value=0
            )

            # Map day numbers to names
            pivot_df.index = [calendar.day_name[d] for d in pivot_df.index]

            # Ensure days are in correct order
            day_order = list(calendar.day_name)
            pivot_df = pivot_df.reindex(day_order)

            # Create heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Hour of Day", y="Day of Week", color="Order Count"),
                x=[f"{h:02d}:00" for h in range(24)],
                y=pivot_df.index,
                color_continuous_scale="Blues",
                title="Order Density Heatmap by Hour and Day"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def iv_to_oral_transition(self):
        """Display IV to oral medication transition analysis."""
        if 'poe' not in self.data or self.data['poe'] is None or 'poe_detail' not in self.data or self.data['poe_detail'] is None:
            st.warning("Provider order entry data not available for IV to oral transition analysis.")
            return

        poe_df = self.data['poe']
        poe_detail_df = self.data['poe_detail']

        st.subheader("IV to Oral Medication Transition Analysis")

        # This is a placeholder for the IV to oral transition analysis
        # The actual implementation would depend on how medications are coded in the dataset
        st.info("""
        IV to oral medication transition analysis requires specific medication coding information in the MIMIC-IV dataset.

        This analysis would typically include:
        1. Identification of IV medications
        2. Identification of corresponding oral medications
        3. Detection of transition patterns from IV to oral formulations
        4. Visualization of transition timing and frequency

        The implementation depends on the specific medication coding and order details available in your MIMIC-IV dataset.
        """)

        # Example visualization (placeholder)
        st.write("#### Example IV to Oral Transition Visualization")

        # Create sample data for demonstration
        np.random.seed(42)
        days = list(range(1, 11))
        iv_counts = [100 - i * 8 + np.random.randint(-5, 5) for i in range(10)]
        oral_counts = [20 + i * 7 + np.random.randint(-3, 3) for i in range(10)]

        # Create dataframe
        example_df = pd.DataFrame({
            'Hospital Day': days,
            'IV Medication': iv_counts,
            'Oral Medication': oral_counts
        })

        # Melt dataframe for plotting
        melted_df = pd.melt(
            example_df,
            id_vars=['Hospital Day'],
            value_vars=['IV Medication', 'Oral Medication'],
            var_name='Route',
            value_name='Count'
        )

        # Create line chart
        fig = px.line(
            melted_df,
            x='Hospital Day',
            y='Count',
            color='Route',
            markers=True,
            color_discrete_map={'IV Medication': '#4682b4', 'Oral Medication': '#e41a1c'},
            title="Example: IV to Oral Transition Pattern"
        )
        fig.update_layout(
            xaxis_title='Hospital Day',
            yaxis_title='Medication Count',
            legend_title='Route'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Example transition rate visualization
        st.write("#### Example Transition Rate by Medication Class")

        # Create sample data for demonstration
        medication_classes = ['Antibiotics', 'Analgesics', 'Antihypertensives', 'Anticoagulants', 'Corticosteroids']
        transition_rates = [78, 65, 42, 58, 35]
        transition_days = [3.2, 2.8, 4.5, 3.7, 5.1]

        # Create dataframe
        example_df2 = pd.DataFrame({
            'Medication Class': medication_classes,
            'Transition Rate (%)': transition_rates,
            'Average Transition Day': transition_days
        })

        # Create bar chart
        fig = px.bar(
            example_df2,
            x='Medication Class',
            y='Transition Rate (%)',
            color='Average Transition Day',
            color_continuous_scale='Blues',
            title="Example: IV to Oral Transition Rate by Medication Class"
        )
        fig.update_layout(
            xaxis_title='Medication Class',
            yaxis_title='Transition Rate (%)',
            coloraxis_colorbar_title='Avg. Transition Day'
        )
        st.plotly_chart(fig, use_container_width=True)

    def sequential_patterns(self):
        """Display sequential pattern mining of orders preceding discharge."""
        if 'poe' not in self.data or self.data['poe'] is None:
            st.warning("Provider order entry data not available for sequential pattern mining.")
            return

        poe_df = self.data['poe']

        if 'ordertime' not in poe_df.columns or 'order_type' not in poe_df.columns:
            st.warning("Order time or type information not available for sequential pattern mining.")
            return

        st.subheader("Sequential Pattern Mining of Orders Preceding Discharge")

        # Pattern mining parameters
        st.write("#### Pattern Mining Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Time window before discharge
            time_window = st.slider(
                "Time Window Before Discharge (hours)",
                min_value=6,
                max_value=72,
                value=24,
                step=6
            )

        with col2:
            # Minimum support threshold
            min_support = st.slider(
                "Minimum Support Threshold",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                format="%.2f"
            )

        # This is a placeholder for the actual sequential pattern mining
        # The actual implementation would depend on the specific data structure and availability
        st.info("""
        Sequential pattern mining requires linking orders to discharge events and analyzing the temporal patterns.

        This analysis would typically include:
        1. Identification of discharge events
        2. Extraction of orders within the specified time window before discharge
        3. Application of sequential pattern mining algorithms
        4. Visualization of frequent patterns and associations

        The implementation depends on the specific order and discharge data available in your MIMIC-IV dataset.
        """)

        # Example visualization (placeholder)
        st.write("#### Example Sequential Patterns Visualization")

        # Create sample data for demonstration
        example_patterns = [
            {'pattern': ['Discharge Order', 'Medication Reconciliation', 'Patient Education'], 'support': 0.42},
            {'pattern': ['Labs Review', 'Discharge Order', 'Medication Reconciliation'], 'support': 0.38},
            {'pattern': ['Vital Signs', 'Labs Review', 'Discharge Order'], 'support': 0.35},
            {'pattern': ['Medication Reconciliation', 'Patient Education', 'Follow-up Scheduling'], 'support': 0.31},
            {'pattern': ['Discharge Order', 'Follow-up Scheduling', 'Discharge Summary'], 'support': 0.28},
            {'pattern': ['Imaging Review', 'Discharge Order', 'Discharge Summary'], 'support': 0.25},
            {'pattern': ['Consult Note', 'Discharge Order', 'Medication Reconciliation'], 'support': 0.22},
            {'pattern': ['Vital Signs', 'Discharge Order', 'Discharge Summary'], 'support': 0.20},
        ]

        # Create dataframe
        example_df = pd.DataFrame(example_patterns)
        example_df['pattern_str'] = example_df['pattern'].apply(lambda x: ' → '.join(x))
        example_df['support_pct'] = example_df['support'] * 100

        # Create bar chart
        fig = px.bar(
            example_df,
            y='pattern_str',
            x='support_pct',
            color='support_pct',
            color_continuous_scale='Blues',
            title="Example: Frequent Order Sequences Before Discharge",
            labels={'pattern_str': 'Order Sequence', 'support_pct': 'Support (%)'},
            orientation='h'
        )
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Support (%)',
            yaxis_title='Order Sequence',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Example network visualization
        st.write("#### Example Order Sequence Network")

        # Create sample data for network visualization
        nodes = [
            {'id': 'Discharge Order', 'group': 1, 'size': 20},
            {'id': 'Medication Reconciliation', 'group': 2, 'size': 15},
            {'id': 'Patient Education', 'group': 2, 'size': 12},
            {'id': 'Labs Review', 'group': 3, 'size': 14},
            {'id': 'Vital Signs', 'group': 3, 'size': 10},
            {'id': 'Follow-up Scheduling', 'group': 2, 'size': 8},
            {'id': 'Discharge Summary', 'group': 1, 'size': 18},
            {'id': 'Imaging Review', 'group': 3, 'size': 7},
            {'id': 'Consult Note', 'group': 4, 'size': 6},
        ]

        links = [
            {'source': 'Labs Review', 'target': 'Discharge Order', 'value': 0.38},
            {'source': 'Vital Signs', 'target': 'Labs Review', 'value': 0.35},
            {'source': 'Discharge Order', 'target': 'Medication Reconciliation', 'value': 0.42},
            {'source': 'Medication Reconciliation', 'target': 'Patient Education', 'value': 0.31},
            {'source': 'Patient Education', 'target': 'Follow-up Scheduling', 'value': 0.31},
            {'source': 'Discharge Order', 'target': 'Follow-up Scheduling', 'value': 0.28},
            {'source': 'Follow-up Scheduling', 'target': 'Discharge Summary', 'value': 0.28},
            {'source': 'Imaging Review', 'target': 'Discharge Order', 'value': 0.25},
            {'source': 'Discharge Order', 'target': 'Discharge Summary', 'value': 0.28},
            {'source': 'Consult Note', 'target': 'Discharge Order', 'value': 0.22},
            {'source': 'Vital Signs', 'target': 'Discharge Order', 'value': 0.20},
        ]

        # Create network graph
        G = nx.DiGraph()

        # Add nodes
        for node in nodes:
            G.add_node(node['id'], group=node['group'], size=node['size'])

        # Add edges
        for link in links:
            G.add_edge(link['source'], link['target'], weight=link['value'])

        # Get node positions using spring layout
        pos = nx.spring_layout(G, seed=42)

        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        edge_width = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[0]} → {edge[1]}: {edge[2]['weight']:.2f}")
            edge_width.append(edge[2]['weight'] * 10)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[0])
            node_size.append(node[1]['size'] * 10)
            node_color.append(node[1]['group'])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Group',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Example: Order Sequence Network",
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                       )

        st.plotly_chart(fig, use_container_width=True)
