#!/usr/bin/env python3
"""
Dask Configuration Optimizer for MIMIC-IV Analysis

This script analyzes your system resources and provides optimal Dask configuration
recommendations for different workload profiles.

Usage:
    python scripts/dask_config_optimizer.py

Requirements:
    pip install psutil
"""

import psutil
import os
import sys
from typing import Dict, Any


class DaskConfigOptimizer:
    """Optimize Dask configuration based on system resources."""
    
    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.memory_percent_used = psutil.virtual_memory().percent
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information."""
        return {
            'cpu_count'          : self.cpu_count,
            'total_memory_gb'    : round(self.total_memory_gb, 1),
            'available_memory_gb': round(self.available_memory_gb, 1),
            'memory_percent_used': round(self.memory_percent_used, 1),
            'free_memory_gb'     : round(self.total_memory_gb - (self.total_memory_gb * self.memory_percent_used / 100), 1)
        }
    
    def get_optimal_balanced_config(self) -> Dict[str, Any]:
        """Get a single optimized configuration that balances all MIMIC-IV workloads."""
        
        # Calculate optimal workers based on system resources
        # Use 50-75% of CPU cores for balanced performance
        optimal_workers = max(2, min(self.cpu_count // 2, 6))
        
        # Calculate optimal threads per worker
        # Balance between I/O and CPU-bound tasks
        optimal_threads = max(4, min(8, 32 // optimal_workers))
        
        # Calculate memory per worker (leave 25% system memory free)
        usable_memory = self.available_memory_gb * 0.75
        memory_per_worker = max(4, int(usable_memory / optimal_workers))
        
        return {
            'n_workers'         : optimal_workers,
            'threads_per_worker': optimal_threads,
            'memory_limit'      : f"{memory_per_worker}GB",
            'total_memory_usage': f"{memory_per_worker * optimal_workers}GB",
            'description'       : 'Optimized balanced configuration for all MIMIC-IV workloads',
            'optimized_for'     : ['data_loading', 'table_merging', 'feature_engineering', 'clustering']
        }
    
    def get_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Generate Dask configuration recommendations."""
        
        # Base calculations
        conservative_workers = max(1, self.cpu_count // 4)
        balanced_workers = max(1, self.cpu_count // 2)
        aggressive_workers = max(1, self.cpu_count - 1)
        
        # Memory calculations (leave 30% system memory free)
        usable_memory = self.available_memory_gb * 0.7
        
        recommendations = {
            'development': {
                'description'       : 'Safe for development and testing with small datasets',
                'use_case'          : 'Small MIMIC-IV subsets, code testing, debugging',
                'n_workers'         : 1,
                'threads_per_worker': 4,
                'memory_limit'      : '4GB',
                'total_memory_usage': '4GB',
                'recommended_for'   : 'Systems with < 16GB RAM or small datasets'
            },
            'conservative': {
                'description'       : 'Stable configuration with low resource usage',
                'use_case'          : 'Standard MIMIC-IV analysis, single table operations',
                'n_workers'         : conservative_workers,
                'threads_per_worker': 8,
                'memory_limit'      : f"{max(4, int(usable_memory / conservative_workers))}GB",
                'total_memory_usage': f"{max(4, int(usable_memory / conservative_workers)) * conservative_workers}GB",
                'recommended_for'   : 'Most users, production environments'
            },
            'balanced': {
                'description'       : 'Good balance between performance and stability',
                'use_case'          : 'Multi-table merging, feature engineering, clustering',
                'n_workers'         : balanced_workers,
                'threads_per_worker': 6,
                'memory_limit'      : f"{max(6, int(usable_memory / balanced_workers))}GB",
                'total_memory_usage': f"{max(6, int(usable_memory / balanced_workers)) * balanced_workers}GB",
                'recommended_for'   : 'Systems with 16-32GB RAM'
            },
            'aggressive': {
                'description'       : 'Maximum performance for large datasets',
                'use_case'          : 'Full MIMIC-IV dataset, intensive computations',
                'n_workers'         : aggressive_workers,
                'threads_per_worker': 2,
                'memory_limit'      : f"{max(8, int(usable_memory / aggressive_workers))}GB",
                'total_memory_usage': f"{max(8, int(usable_memory / aggressive_workers)) * aggressive_workers}GB",
                'recommended_for'   : 'High-end systems with 32GB+ RAM'
            }
        }
        
        return recommendations
    
    def get_workload_specific_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations optimized for specific MIMIC-IV workloads."""
        
        base_memory = max(4, int(self.available_memory_gb * 0.15))
        
        return {
            'data_loading': {
                'description'       : 'Optimized for reading large CSV/Parquet files',
                'n_workers'         : max(1, self.cpu_count // 3),
                'threads_per_worker': 16,
                'memory_limit'      : f"{base_memory}GB",
                'notes'             : 'High thread count for I/O operations'
            },
            'table_merging': {
                'description'       : 'Optimized for joining multiple MIMIC-IV tables',
                'n_workers'         : max(2, self.cpu_count // 2),
                'threads_per_worker': 4,
                'memory_limit'      : f"{base_memory * 2}GB",
                'notes'             : 'Balanced for shuffle-heavy operations'
            },
            'feature_engineering': {
                'description'       : 'Optimized for complex transformations and calculations',
                'n_workers'         : max(2, self.cpu_count - 1),
                'threads_per_worker': 2,
                'memory_limit'      : f"{base_memory}GB",
                'notes'             : 'CPU-optimized for computational tasks'
            },
            'clustering_analysis': {
                'description'       : 'Optimized for machine learning and clustering',
                'n_workers'         : max(1, self.cpu_count // 2),
                'threads_per_worker': 4,
                'memory_limit'      : f"{base_memory * 3}GB",
                'notes'             : 'High memory for algorithm requirements'
            }
        }
    
    def validate_config(self, n_workers: int, threads_per_worker: int, memory_limit: str) -> Dict[str, Any]:
        """Validate a Dask configuration against system resources."""
        
        # Parse memory limit
        memory_gb          = float(memory_limit.replace('GB', '').replace('gb', ''))
        total_memory_usage = n_workers * memory_gb
        total_threads      = n_workers * threads_per_worker
        
        warnings = []
        errors   = []
        
        # Memory validation
        if total_memory_usage > self.available_memory_gb * 0.8:
            errors.append(f"Total memory usage ({total_memory_usage}GB) exceeds 80% of available memory ({self.available_memory_gb:.1f}GB)")
        elif total_memory_usage > self.available_memory_gb * 0.6:
            warnings.append(f"High memory usage ({total_memory_usage}GB) - monitor for stability")
        
        # CPU validation
        if total_threads > self.cpu_count * 2:
            warnings.append(f"Total threads ({total_threads}) exceeds 2x CPU cores ({self.cpu_count}) - may cause context switching overhead")
        
        # Worker validation
        if n_workers > self.cpu_count:
            warnings.append(f"More workers ({n_workers}) than CPU cores ({self.cpu_count}) - may reduce efficiency")
        
        # Memory per worker validation
        if memory_gb < 2:
            errors.append(f"Memory per worker ({memory_gb}GB) is too low - minimum 2GB recommended")
        
        return {
            'valid': len(errors) == 0,
            'total_memory_usage_gb': total_memory_usage,
            'total_threads': total_threads,
            'memory_utilization_percent': (total_memory_usage / self.available_memory_gb) * 100,
            'warnings': warnings,
            'errors': errors
        }
    
    def print_system_info(self):
        """Print detailed system information."""
        info = self.get_system_info()
        
        print("=" * 60)
        print("SYSTEM RESOURCE ANALYSIS")
        print("=" * 60)
        print(f"CPU Cores: {info['cpu_count']}")
        print(f"Total Memory: {info['total_memory_gb']} GB")
        print(f"Available Memory: {info['available_memory_gb']} GB")
        print(f"Memory Usage: {info['memory_percent_used']}%")
        print(f"Free Memory: {info['free_memory_gb']} GB")
        print()
    
    def print_recommendations(self):
        """Print configuration recommendations."""
        recommendations = self.get_recommendations()
        
        print("=" * 60)
        print("DASK CONFIGURATION RECOMMENDATIONS")
        print("=" * 60)
        
        for profile_name, config in recommendations.items():
            print(f"\n📊 {profile_name.upper()} PROFILE")
            print(f"   Description: {config['description']}")
            print(f"   Use Case: {config['use_case']}")
            print(f"   Recommended For: {config['recommended_for']}")
            print("   ")
            print("   Configuration:")
            print(f"     n_workers = {config['n_workers']}")
            print(f"     threads_per_worker = {config['threads_per_worker']}")
            print(f"     memory_limit = '{config['memory_limit']}'")
            print("   ")
            print(f"   Total Memory Usage: {config['total_memory_usage']}")
            
            # Validate configuration
            validation = self.validate_config(
                config['n_workers'], 
                config['threads_per_worker'], 
                config['memory_limit']
            )
            
            if validation['warnings']:
                print(f"   ⚠️  Warnings: {'; '.join(validation['warnings'])}")
            if validation['errors']:
                print(f"   ❌ Errors: {'; '.join(validation['errors'])}")
            else:
                print(f"   ✅ Configuration is valid")
    
    def print_workload_configs(self):
        """Print workload-specific configurations."""
        configs = self.get_workload_specific_configs()
        
        print("\n" + "=" * 60)
        print("WORKLOAD-SPECIFIC CONFIGURATIONS")
        print("=" * 60)
        
        for workload_name, config in configs.items():
            print(f"\n🎯 {workload_name.upper().replace('_', ' ')}")
            print(f"   Description: {config['description']}")
            print(f"   Configuration:")
            print(f"     n_workers = {config['n_workers']}")
            print(f"     threads_per_worker = {config['threads_per_worker']}")
            print(f"     memory_limit = '{config['memory_limit']}'")
            print(f"   Notes: {config['notes']}")
    
    def generate_streamlit_config(self, profile: str = 'balanced') -> str:
        """Generate Streamlit app configuration code."""
        recommendations = self.get_recommendations()
        
        if profile not in recommendations:
            profile = 'balanced'
        
        config = recommendations[profile]
        
        return f"""
            # Add this to your Streamlit app configuration
            # In mimic_iv_analysis/visualization/app.py, update the _dask_configuration method defaults:

            if 'dask_n_workers' not in st.session_state:
                st.session_state.dask_n_workers = {config['n_workers']}
            if 'dask_threads_per_worker' not in st.session_state:
                st.session_state.dask_threads_per_worker = {config['threads_per_worker']}
            if 'dask_memory_limit' not in st.session_state:
                st.session_state.dask_memory_limit = '{config['memory_limit']}'
            """
    
    @staticmethod
    def get_optimized_config_for_streamlit() -> Dict[str, Any]:
        """Static method to get optimized configuration for Streamlit integration."""
        optimizer = DaskConfigOptimizer()
        config = optimizer.get_optimal_balanced_config()
        
        return {
            'n_workers'         : config['n_workers'],
            'threads_per_worker': config['threads_per_worker'],
            'memory_limit'      : config['memory_limit'],
            'description'       : config['description']
        }


def main():
    """Main function to run the optimizer."""
    
    try:
        optimizer = DaskConfigOptimizer()
        
        # Print system information
        optimizer.print_system_info()
        
        # Print recommendations
        optimizer.print_recommendations()
        
        # Print workload-specific configs
        optimizer.print_workload_configs()
        
        # Generate Streamlit configuration
        print("\n" + "=" * 60)
        print("STREAMLIT APP CONFIGURATION")
        print("=" * 60)
        print(optimizer.generate_streamlit_config('balanced'))
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Choose a configuration profile based on your use case")
        print("2. Update the Dask configuration in your Streamlit app")
        print("3. Monitor performance using the Dask dashboard (http://localhost:8787)")
        print("4. Adjust parameters based on actual workload performance")
        print("\nFor detailed guidance, see: documentations/DASK_OPTIMIZATION_GUIDE.md")
        
    except ImportError:
        print("Error: psutil package is required. Install with: pip install psutil")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()