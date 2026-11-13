"""
Configuration loader for BaGGLS project.
Handles loading of configuration files and path resolution.
"""
import yaml
from pathlib import Path
import os

def get_project_root():
    """Get the project root directory."""
    # Assumes this file is in src/utils/
    return Path(__file__).parent.parent.parent

def load_config(config_name):
    """
    Load a configuration file from the configs directory.
    
    Args:
        config_name: Name of config file (with or without .yml extension)
    
    Returns:
        dict: Configuration dictionary
    """
    if not config_name.endswith('.yml'):
        config_name += '.yml'
    
    project_root = get_project_root()
    config_path = project_root / 'configs' / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def resolve_path(path_str, base_path=None):
    """
    Resolve a path string to an absolute path.
    
    Args:
        path_str: Path string (can be relative or absolute)
        base_path: Base path for relative paths (default: project root)
    
    Returns:
        Path: Resolved absolute path
    """
    if path_str is None:
        return None
    
    path = Path(path_str)
    
    # If already absolute, return as is
    if path.is_absolute():
        return path
    
    # Otherwise, resolve relative to base_path
    if base_path is None:
        base_path = get_project_root()
    
    return (Path(base_path) / path).resolve()

class PathConfig:
    """Container for all project paths with lazy resolution."""
    
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = load_config('paths')
        
        self.config = config_dict
        self.project_root = get_project_root()
        self.data_root = resolve_path(config_dict.get('data_root', 'data'), self.project_root)
    
    def get_path(self, key, mkdir=False):
        """
        Get a resolved path from the config.
        
        Args:
            key: Key in the config (can be nested with dots, e.g., 'external.jaspar_motif_file')
            mkdir: If True, create the directory if it doesn't exist
        
        Returns:
            Path: Resolved path
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        
        if value is None:
            return None
        
        # Handle special cases
        if key in ['testdata_dir', 'xai_dir', 'vb_models_dir']:
            path = resolve_path(value, self.data_root)
        elif key == 'external.jaspar_motif_file':
            # JASPAR file is relative to data_root
            path = resolve_path(value, self.data_root)
        else:
            path = resolve_path(value, self.project_root)
        
        if mkdir and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return path