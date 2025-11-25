import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

class ProfileManager:
    """
    Industrial-grade loader for heterogeneous load profiles.
    Supports CSV, Parquet, and NPY formats.
    Handles time alignment and memory-mapped loading for large datasets.
    """
    def __init__(self, time_index: pd.DatetimeIndex):
        self.sim_index = time_index
        self.n_steps = len(time_index)

    def load_aggregated_csv(self, file_path: str, col_map: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Loads a single CSV containing columns for different profiles (e.g., 'price', 'carbon').
        """
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        df = self._align_dataframe(df)
        
        data = {}
        for file_col, internal_key in col_map.items():
            if file_col not in df.columns:
                raise ValueError(f"Column '{file_col}' not found in {file_path}")
            data[internal_key] = df[file_col].values.astype(np.float32)
        return data

    def load_batch_profiles(
        self, 
        source: Union[str, np.ndarray], 
        n_houses: int, 
        expected_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Loads a matrix of profiles (N_houses, TimeSteps).
        
        Args:
            source: Path to .npy file (recommended for speed) or .parquet.
            n_houses: Number of profiles to slice.
        """
        if isinstance(source, np.ndarray):
            return source[:n_houses].astype(np.float32)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")

        if path.suffix == '.npy':
            # Memory map for instant access without loading everything to RAM
            # Assumes shape is (Total_Houses, Total_Time)
            arr = np.load(path, mmap_mode='r')
            
            # Validate time dimension
            if arr.shape[1] != self.n_steps:
                # In production, we might auto-resample here, but for now, fail fast
                raise ValueError(f"Profile time dimension mismatch. Expected {self.n_steps}, got {arr.shape[1]}")
                
            # Slice the requested batch
            return np.array(arr[:n_houses], dtype=np.float32)
            
        elif path.suffix == '.parquet':
            # Parquet is column-oriented. Assume columns are House_IDs, Index is Time.
            df = pd.read_parquet(path)
            df = self._align_dataframe(df)
            # Transpose to (N_houses, Time)
            return df.iloc[:, :n_houses].values.T.astype(np.float32)
        
        else:
            raise ValueError("Production loader requires .npy (fast) or .parquet (compressed). CSV is too slow for batch profiles.")

    def _align_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindexes dataframe to match the simulation clock exactly."""
        # Handle missing timezones or mismatches
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # Resample if needed (e.g. 1H -> 15min)
        if len(df) != self.n_steps:
            # Simple upsample via linear interp
            df = df.resample(pd.to_timedelta(self.sim_index.freq)).interpolate(method='linear')
            
        # Strict reindex to handle missing/extra rows
        df = df.reindex(self.sim_index, fill_value=0.0).ffill().bfill()
        return df