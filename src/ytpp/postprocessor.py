"""
PeleLMeX Postprocessor - A unified interface for post-processing PeleLMeX simulation data.

Supports both 2D and 3D AMReX/BoxLib data formats.
"""

import os
from typing import Optional, Union
from unittest import case

import numpy as np
import pandas as pd
import yt
import matplotlib.pyplot as plt
from PIL import Image

yt.set_log_level(40)


class PeleLMeXPostProcessor:
    """
    Post-processing class for PeleLMeX simulation data (supports 2D and 3D).

    Provides unified interface for:
    - Point sampling
    - Line sampling
    - Slice extraction
    - Volume rendering (3D only)

    Parameters
    ----------
    path_pattern : str
        Path pattern for loading data (supports wildcards like './plt3d*' or './plt2d/plt*')
    units_override : dict, optional
        Units override dictionary. Default: {"length_unit": (1, "m")}
    twoD : bool, optional
        Flag indicating if the data is 2D. Default: False
        Note that for 2D cases, PeleLMeX outputs 3D data with the z direction ranging from 0 to 1
    output_dir : str, optional
        Base output directory for plots. Default: "./pp"

    Examples
    --------
    >>> pp = PeleLMeXPostProcessor("./plt3d*")  # 3D case
    >>> pp = PeleLMeXPostProcessor("./plt2d/plt*")  # 2D case
    >>> print(pp.domain_left, pp.domain_right)
    >>> data = pp.point_sample([0.001, 0.015, 0.0001], time=0.07)  # 3D
    >>> data = pp.point_sample([0.001, 0.015], time=0.07)  # 2D
    """

    def __init__(self, path_pattern: str, units_override: dict = None, twoD: bool = False,
                 output_dir: str = "./pp"):
        if units_override is None:
            units_override = {"length_unit": (1, "m")}

        self._ts = yt.load(path_pattern, units_override=units_override)
        self._units_override = units_override
        self._output_dir = output_dir

        # Cache metadata from first dataset
        ds = self._ts[0]
        self._domain_left = np.array(ds.domain_left_edge.to('m').value)
        self._domain_right = np.array(ds.domain_right_edge.to('m').value)
        self._domain_width = np.array(ds.domain_width.to('m').value)
        self._domain_center = np.array(ds.domain_center.to('m').value)
        
        # 2D flag
        self._twoD = twoD

        # Use derived_field_list for more fields
        self._times = [float(ds.current_time) for ds in self._ts]
        self._field_family_names, self._field_list, self._field_pair_list = self.constructFullFieldList()


    def constructFullFieldList(self):
        """
        Build the complete field list from the dataset's derived fields.

        Returns
        -------
        tuple
            (field_family_names, field_full_list, field_pair_list) where:
            - field_family_names: dict mapping field_name -> field_family (e.g., 'temp' -> 'boxlib')
            - field_full_list: list of all unique field names
            - field_pair_list: list of (field_family, field_name) tuples
        """
        full_field_names = self._ts[0].derived_field_list
        field_family_names = {}
        for f in full_field_names:
            if f[1] not in field_family_names:
                field_family_names[f[1]] = f[0]
        field_full_list = field_family_names.keys()

        field_pair_list = []
        for f in field_full_list:
            field_pair_list.append((field_family_names[f], f))
        return field_family_names, field_full_list, field_pair_list

    # =========================================================================
    # Metadata Properties
    # =========================================================================

    @property
    def domain_left(self) -> np.ndarray:
        """Domain left boundary coordinates in meters."""
        return self._domain_left

    @property
    def domain_right(self) -> np.ndarray:
        """Domain right boundary coordinates in meters."""
        return self._domain_right

    @property
    def domain_width(self) -> np.ndarray:
        """Domain width in meters."""
        return self._domain_width

    @property
    def domain_center(self) -> np.ndarray:
        """Domain center coordinates in meters."""
        return self._domain_center

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps."""
        return len(self._ts)

    @property
    def field_list(self) -> list:
        """List of available field names (derived fields)."""
        return self._field_list

    @property
    def times(self) -> list:
        """List of time values for all timesteps."""
        return self._times

    def print_mesh_info(self, time=None) -> None:
        """Print mesh statistics for the specified timestep."""
        ds = self._resolve_timestep(time)
        print(f"Dataset: {ds.basename}, t = {float(ds.current_time):.6f} s")
        print(f"Domain: {ds.domain_left_edge.to('m')} -> {ds.domain_right_edge.to('m')}")
        ds.print_stats()


    def num_cells(self, time=None, per_level: bool = False):
        """
        Return the number of cells for the specified timestep.

        Counts all stored cells across every grid, including coarse cells
        hidden under finer levels, matching print_mesh_info()/print_stats()
        (e.g. 39424). Note this counts cells, not grid boxes (see num_grids()).

        Parameters
        ----------
        time : float or None
            Time value to find the closest timestep, or None for the last.
        per_level : bool, optional
            If True, return a numpy array of counts per level (0 .. max_level)
            instead of the total.

        Returns
        -------
        int or np.ndarray
            Total cell count, or per-level counts if per_level=True.
        """
        ds = self._resolve_timestep(time)
        max_level = ds.index.max_level
        counts = np.asarray(ds.index.level_stats["numcells"][:max_level + 1], dtype=int)
        if per_level:
            return counts
        return int(counts.sum())

    def base_grid(self, time=None) -> np.ndarray:
        """
        Return the base (level-0) grid dimensions in cells per axis.

        These are the logical level-0 cell counts spanning the whole domain
        (e.g. [32, 64, 1]), i.e. the resolution of the coarsest level, not the
        number of grid boxes (that is num_grids()).

        Parameters
        ----------
        time : float or None
            Time value to find the closest timestep, or None for the last.

        Returns
        -------
        np.ndarray
            Level-0 cell counts per axis.
        """
        ds = self._resolve_timestep(time)
        return np.asarray(ds.domain_dimensions)

    def get_dataset(self, time=None):
        """
        Get the yt dataset for a specified timestep.

        Parameters
        ----------
        time : float or None
            Time value to find the closest timestep, or None for the last timestep

        Returns
        -------
        yt.Dataset
            The dataset for the specified timestep
        """
        return self._resolve_timestep(time)

    def suggest_camera_settings(self) -> None:
        """Print suggested camera settings for volume rendering (3D only)."""
        center = self._domain_center
        width = self._domain_width

        print("=== Suggested Camera Settings for Volume Rendering ===")
        print(f"\nDomain center (focus): {center} m")
        print(f"Domain width: {width} m")
        print("\nSuggested camera positions:")
        print(f"  View from -X: [{center[0] - 3*width[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
        print(f"  View from +X: [{center[0] + 3*width[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
        print(f"  View from -Y: [{center[0]:.4f}, {center[1] - 3*width[1]:.4f}, {center[2]:.4f}]")
        print(f"  View from +Y: [{center[0]:.4f}, {center[1] + 3*width[1]:.4f}, {center[2]:.4f}]")
        print(f"  View from -Z: [{center[0]:.4f}, {center[1]:.4f}, {center[2] - 3*width[2]:.4f}]")
        print(f"  View from +Z: [{center[0]:.4f}, {center[1]:.4f}, {center[2] + 3*width[2]:.4f}]")
        print(f"  Diagonal view: [{center[0] - 2*width[0]:.4f}, {center[1] - 2*width[1]:.4f}, {center[2] + 2*width[2]:.4f}]")
        print("\nSuggested focus: domain_center (default)")
        print("Suggested rotation: 0 (no rotation), np.pi/2 (90°), np.pi (180°)")

    # =========================================================================
    # Time Resolution
    # =========================================================================

    def _resolve_timestep(self, time):
        """
        Resolve time specification to a yt dataset.

        Parameters
        ----------
        time : float or None
            - None: last timestep
            - float/int: find closest timestep to this time value

        Returns
        -------
        yt.Dataset
            The resolved dataset
        """
        if time is None:
            return self._ts[-1]

        # Find closest timestep to the given time value
        times_arr = np.array(self._times)
        idx = np.argmin(np.abs(times_arr - time))
        return self._ts[idx]

    def _get_timestep_index(self, time) -> int:
        """
        Get the index of the resolved timestep.

        Parameters
        ----------
        time : float or None
            Time value to find. None returns the last index.

        Returns
        -------
        int
            Index of the closest timestep
        """
        if time is None:
            return len(self._ts) - 1
        times_arr = np.array(self._times)
        return int(np.argmin(np.abs(times_arr - time)))

    def _filter_times(self, time_range: tuple = None) -> dict:
        """
        Filter timestep indices based on time range.

        Parameters
        ----------
        time_range : tuple, optional
            (start_time, end_time) to filter. None means all times.

        Returns
        -------
        dictionary
            dictionary of (index, time) tuples within range
        """
        
        tempDict = {}
        if time_range is None:
            for i, t in enumerate(self._times):
                tempDict[i] = t
            return tempDict

        start, end = time_range
        for i, t in enumerate(self._times):
            if start-1e-10 <= t <= end+1e-10:
                tempDict[i] = t
        return tempDict

    # =========================================================================
    # Output Path Generation
    # =========================================================================

    def _generate_output_path(self, plot_type: str, name: str, field: str,
                              time: float) -> str:
        """
        Generate output path for plot files.

        Format: {output_dir}/{plot_type}/{name}.{field}.t{time:.2g}.png

        Parameters
        ----------
        plot_type : str
            Type of plot ('vol')
        name : str
            User-provided name for the plot
        field : str
            Field name
        time : float
            Time value

        Returns
        -------
        str
            Full output path
        """
        # Format time with 2 significant digits
        if time == 0:
            time_str = "0.00"
        else:
            time_str = f"{time:.2g}"

        # Create directory
        dir_path = os.path.join(self._output_dir, plot_type)
        os.makedirs(dir_path, exist_ok=True)

        # Generate filename
        filename = f"{name}.{field}.t{time_str}.png"
        return os.path.join(dir_path, filename)

    # =========================================================================
    # Field Handling
    # =========================================================================

    def _normalize_fields(self, fields: list = None, full_name = False) -> list:
        """
        Normalize field names to full yt field tuples.

        Parameters
        ----------
        fields : list, optional
            List of field names (e.g., ["temp", "Y(N2)"])
            None means all available fields
        full_name : bool, optional
            Currently unused, reserved for future use

        Returns
        -------
        list
            List of (field_family, field_name) tuples, e.g., ('boxlib', 'temp') or ('gas', 'velocity_x')
        """
        if fields is None:
            return self._field_pair_list
        else:
            field_tuples = []
            for f in fields:
                if f in self._field_family_names:
                    field_tuples.append((self._field_family_names[f], f))
            return field_tuples

    # =========================================================================
    # Point Sampling
    # =========================================================================

    def point_sample(self, position: list, fields: list = None,
                     time=None) -> pd.Series:
        """
        Sample data at a single point.

        Parameters
        ----------
        position : list
            Coordinates in meters: [x, y] for 2D, [x, y, z] for 3D
        fields : list, optional
            List of field names to sample. None for all fields.
        time : float, int, or None
            Time specification

        Returns
        -------
        pd.Series
            Series with field names as index
        """
        ds = self._resolve_timestep(time)
        if (self._twoD and len(position) == 2):
            position = [position[0], position[1], 0.5]  # Middle of z range for 2D
            
        # perform dimension check
        self._check_dimension(position, 3)
        
        # sample point data
        pos = ds.arr(position, "m")
        probe = ds.point(pos)

        field_tuples = self._normalize_fields(fields)

        data = {}
        for ft in field_tuples:
            vals = probe[ft]
            data[ft[1]] = float(vals[0]) if len(vals) > 0 else np.nan

        return pd.Series(data)

    def point_sample_all_times(self, position: list, fields: list = None,
                               time_range: tuple = None) -> dict:
        """
        Sample data at a single point across all timesteps.

        Parameters
        ----------
        position : list
            Coordinates in meters: [x, y] for 2D, [x, y, z] for 3D
        fields : list, optional
            List of field names to sample
        time_range : tuple, optional
            (start_time, end_time) to filter timesteps

        Returns
        -------
        dict
            {time: pd.Series} dictionary
        """
        filtered = self._filter_times(time_range)
        results = {}

        for idx, t in filtered.items():
            time_value = float(filtered[idx])
            results[t] = self.point_sample(position, fields, time=time_value)

        return results

    # =========================================================================
    # Line Sampling
    # =========================================================================

    def line_sample(self, start: list, end: list, fields: list = None,
                    time=None) -> pd.DataFrame:
        """
        Sample data along a line (3D only, or 2D with fixed z).

        Parameters
        ----------
        start : list
            [x, y, z] start coordinates in meters
        end : list
            [x, y, z] end coordinates in meters
        fields : list, optional
            List of field names. None for all fields.
        time : float, int, or None
            Time specification

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ["x", "y", "z", "t", field1, field2, ...]
        """
        ds = self._resolve_timestep(time)
        current_time = float(ds.current_time)
        
        if (self._twoD and len(start) == 2 ):
            start = [start[0], start[1], 0.5]  # Middle of z range for 2D
        if (self._twoD and len(end) == 2 ):
            end = [end[0], end[1], 0.5]  # Middle of z range for 2D
            
            
        # perform dimension check
        self._check_dimension(start, 3)
        self._check_dimension(end, 3)

        start_pt = ds.arr(start, "m")
        end_pt = ds.arr(end, "m")
        ray = ds.ray(start_pt, end_pt)

        # Get coordinates
        x = np.array(ray["boxlib", "x"].to("m"))
        y = np.array(ray["boxlib", "y"].to("m"))
        z = np.array(ray["boxlib", "z"].to("m"))
        t_arr = np.full_like(x, current_time)

        data = {"x": x, "y": y, "z": z, "t": t_arr}

        # Get field data
        field_tuples = self._normalize_fields(fields)
        for ft in field_tuples:
            data[ft[1]] = np.array(ray[ft])

        df = pd.DataFrame(data)

        # Sort by distance along ray
        df["_dist"] = np.sqrt((df["x"] - start[0])**2 +
                              (df["y"] - start[1])**2 +
                              (df["z"] - start[2])**2)
        df = df.sort_values("_dist").drop("_dist", axis=1).reset_index(drop=True)

        return df

    def line_sample_all_times(self, start: list, end: list, fields: list = None,
                              time_range: tuple = None) -> dict:
        """
        Sample data along a line across all timesteps (3D only).

        Parameters
        ----------
        start : list
            [x, y, z] start coordinates in meters
        end : list
            [x, y, z] end coordinates in meters
        fields : list, optional
            List of field names
        time_range : tuple, optional
            (start_time, end_time) to filter timesteps

        Returns
        -------
        dict
            {time: pd.DataFrame} dictionary
        """
        filtered = self._filter_times(time_range)
        results = {}

        for idx, t in filtered.items():
            results[f"{t:.3e}"] = self.line_sample(start, end, fields, time=float(filtered[idx]))

        return results

    # =========================================================================
    # Slice
    # =========================================================================

    def slice(self, axis: str, position: float, fields: list = None,
              time=None) -> pd.DataFrame:
        """
        Extract a 2D slice at specified position (3D only).

        For 2D data, use the dedicated 2D plotting methods instead.

        Parameters
        ----------
        axis : str
            Slice normal axis ('x', 'y', or 'z')
        position : float
            Position along axis in meters
        fields : list, optional
            List of field names. None for all fields.
        time : float, int, or None
            Time specification

        Returns
        -------
        pd.DataFrame
            DataFrame with coordinate columns and field columns
        """
        ds = self._resolve_timestep(time)
        current_time = float(ds.current_time)
        
        # Override for 2D case - always slice through z at midpoint
        if self._twoD:
            axis = "z"
            position = 0.5  # Middle of z range for 2D data
            coord1, coord2 = 'x', 'y'
        else:
            # Determine coordinate names based on slice axis
            axis_map = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}
            coord1, coord2 = axis_map[axis]

        slc = ds.slice(axis, ds.quan(position, 'm'))
            

        # Get coordinates
        c1 = np.array(slc[('index', coord1)].to('m'))
        c2 = np.array(slc[('index', coord2)].to('m'))

        data = {coord1: c1, coord2: c2}

        # Get field data; convert cell-size length fields to meters like coords
        length_fields = {"dx", "dy", "dz"}
        field_tuples = self._normalize_fields(fields)
        for ft in field_tuples:
            if ft[1] in length_fields:
                data[ft[1]] = np.array(slc[ft].to("m"))
            else:
                data[ft[1]] = np.array(slc[ft])

        df = pd.DataFrame(data)

        return df

    def slice_all_times(self, axis: str, position: float, fields: list = None,
                        time_range: tuple = None) -> dict:
        """
        Extract slices across all timesteps (3D only).

        Parameters
        ----------
        axis : str
            Slice normal axis ('x', 'y', or 'z')
        position : float
            Position along axis in meters
        fields : list, optional
            List of field names
        time_range : tuple, optional
            (start_time, end_time) to filter timesteps

        Returns
        -------
        dict
            {time: pd.DataFrame} dictionary
        """
        filtered = self._filter_times(time_range)
        results = {}

        for idx, t in filtered.items():
            time_value = float(filtered[idx])   
            results[t] = self.slice(axis, position, fields, time=time_value)

        return results

    # =========================================================================
    # Full Domain Data Extraction
    # =========================================================================

    def to_dataframe(self, fields: list = None, time=None,
                     include_coords: bool = True, all_fields = False) -> pd.DataFrame:
        """
        Extract complete field data for a single timestep as a pandas DataFrame.

        Each row represents one grid cell, columns contain spatial coordinates
        and field values.

        Parameters
        ----------
        fields : list, optional
            List of field names to extract (e.g., ["temp", "Y(N2O)", "density"]).
            None extracts all available fields.
        time : float or None, optional
            Time specification:
            - None: last timestep (default)
            - float: closest timestep to this time value
        include_coords : bool, optional
            If True (default), include coordinate columns (x, y, z, t).
            If False, only include field data columns.
        all_fields : bool, optional
            If True, extract all native (boxlib) fields, ignoring the fields parameter.
            Derived yt fields (gas, index) are excluded to avoid ghost cell errors
            on non-periodic boundaries. Default: False.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - Coordinates (if include_coords=True): x, y, z (omitted for 2D), t
            - Fields: one column per field

        Notes
        -----
        - For large 3D datasets, this may consume significant memory.
        - All spatial coordinates are in meters (SI units).
        - AMR grid resolution is preserved (mixed cell sizes).
        - For 2D data, z coordinate column is omitted.

        Examples
        --------
        >>> case = PeleLMeXPostProcessor("plt*", twoD=True)
        >>> df = case.to_dataframe(fields=["temp", "density"], time=0.003)
        >>> df_all = case.to_dataframe(all_fields=True, time=0.003)
        >>> df_no_coords = case.to_dataframe(fields=["temp"], include_coords=False)
        """
        # Resolve timestep
        ds = self._resolve_timestep(time)
        current_time = float(ds.current_time)

        # Get all data object
        ad = ds.all_data()

        if (include_coords):
            if ("x" not in fields):
                fields.append("x")
            if ("y" not in fields):
                fields.append("y")
            if (not self._twoD and "z" not in fields):
                fields.append("z")

        # Normalize field names
        field_tuples = self._normalize_fields(fields)

        if all_fields:
            # Only use native boxlib fields, skip yt derived fields (gas, index)
            # which may fail on non-periodic boundaries due to ghost cell computation
            field_tuples = [ft for ft in self._field_pair_list if (ft[0] == 'boxlib')]

        # Build data dictionary
        data = {}

        # Cell sizes are length fields; convert them to meters like the
        # coordinates so the DataFrame is unit-consistent.
        length_fields = {"dx", "dy", "dz"}

        # Add field data
        for ft in field_tuples:
            if ft[1] in length_fields:
                data[ft[1]] = np.array(ad[ft].to("m"))
            else:
                data[ft[1]] = np.array(ad[ft])

        # add/overwrite coordinates
        if ("x" in fields):
            data['x'] = np.array(ad[('index', 'x')].to('m'))
        if ("y" in fields):
            data['y'] = np.array(ad[('index', 'y')].to('m'))
        if (not self._twoD and "z" in fields):
            data['z'] = np.array(ad[('index', 'z')].to('m'))

        if include_coords:
            data['t'] = np.full(len(data['x']), current_time)


        # Create DataFrame
        df = pd.DataFrame(data)

        return df

    def to_dataframe_all_times(self, fields: list = None,
                               time_range: tuple = None,
                               include_coords: bool = True, all_fields = False) -> dict:
        """
        Extract complete field data for multiple timesteps as a dictionary of DataFrames.

        Parameters
        ----------
        fields : list, optional
            List of field names to extract.
            None extracts all available fields.
        time_range : tuple, optional
            (start_time, end_time) to filter timesteps.
            None includes all timesteps.
        include_coords : bool, optional
            If True (default), include coordinate columns in each DataFrame.
        all_fields : bool, optional
            If True, extract all native (boxlib) fields. Default: False.

        Returns
        -------
        dict
            Dictionary mapping {time_value: pd.DataFrame}.
            Each DataFrame has the same structure as to_dataframe() output.

        Notes
        -----
        - Memory usage scales with number of timesteps x grid size x number of fields.
        - For large datasets, consider processing timesteps individually.

        Examples
        --------
        >>> case = PeleLMeXPostProcessor("plt*", twoD=True)
        >>> dfs = case.to_dataframe_all_times(
        ...     fields=["temp", "Y(N2O)"],
        ...     time_range=(0.0, 0.004)
        ... )
        >>> for time_val, df in dfs.items():
        ...     print(f"Time {time_val:.4f} s: {len(df)} cells")
        """
        # Filter timesteps
        filtered = self._filter_times(time_range)
        results = {}


        # Extract data for each timestep
        for idx, t in filtered.items():
            time_value = float(filtered[idx])
            results[t] = self.to_dataframe(
                fields=fields,
                time=time_value,
                include_coords=include_coords,
                all_fields=all_fields
            )

        return results

    # =========================================================================
    # AMR -> uniform finest grid (for imshow)
    # =========================================================================

    @staticmethod
    def _fill_blocks(matrix, vals, mask, i0, i1, n) -> None:
        """Write each masked cell's value into its n x n block of the finest grid."""
        base0 = i0[mask]
        base1 = i1[mask]
        v = vals[mask]
        for a in range(n):        # offset along the 1st (horizontal) axis
            for b in range(n):    # offset along the 2nd (vertical) axis
                matrix[base1 + b, base0 + a] = v

    @staticmethod
    def expand_df_to_matrix(df: pd.DataFrame, field: str, plane: str = "xy",
                            dx_col: str = "dx") -> tuple:
        """
        Resample AMR cells from a slice DataFrame onto the uniform finest grid.

        Everything is derived from the DataFrame: the finest cell size is
        dx_min = df[dx_col].min(), the grid extent is the cell edges
        (center +/- dx/2), and the resolution is that extent divided by
        dx_min. Every cell is tiled over the (dx / dx_min) ** 2 finest cells it
        covers, each inheriting the cell's value. Coordinates and dx_col must be
        in the same units (to_dataframe returns both in meters). Cells are
        assumed cubic (dx == dy == dz), as in PeleLMeX.

        Parameters
        ----------
        df : pd.DataFrame
            Slice data with coordinate columns, a `dx_col` column and `field`.
            Get it from to_dataframe(fields=[field, "dx"], ...) for 2D, or from
            a single xy/xz/yz cut for 3D.
        field : str
            Name of the field column to resample.
        plane : str
            Coordinate pair to map: 'xy', 'xz' or 'yz'. Use 'xy' for 2D data;
            for 3D pass the df of the desired cut and the matching plane.
        dx_col : str
            Column holding each cell's width, in the same units as the
            coordinates. Default 'dx'.

        Returns
        -------
        matrix : np.ndarray
            Field values on the finest grid, shape (n1, n0): row = 2nd axis,
            col = 1st axis. Ready for imshow(origin='lower').
        c0 : np.ndarray
            Cell-center coordinates along the 1st (horizontal) axis, length n0.
        c1 : np.ndarray
            Cell-center coordinates along the 2nd (vertical) axis, length n1.
        """
        # the cell width column is required to size and place the blocks
        if dx_col not in df.columns:
            raise ValueError(
                f"df must contain a '{dx_col}' column; "
                f"e.g. to_dataframe(fields=['{field}', 'dx'], ...)")

        # pick the two coordinate columns for the requested plane
        cols = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("z", "y")}
        col0, col1 = cols[plane]

        # source cell centers, widths and values
        p0 = df[col0].to_numpy()
        p1 = df[col1].to_numpy()
        dx = df[dx_col].to_numpy()
        vals = df[field].to_numpy()

        # finest cell size and each cell's block size (cells to render per side)
        fine = dx.min()
        counts = np.round(dx / fine).astype(int)

        # grid extent from cell edges (center +/- half width), then resolution
        half = dx / 2
        left0 = (p0 - half).min()
        left1 = (p1 - half).min()
        n0 = int(round(((p0 + half).max() - left0) / fine))
        n1 = int(round(((p1 + half).max() - left1) / fine))

        # lower-left finest-cell index of each cell's block
        i0 = np.round((p0 - left0) / fine - counts / 2).astype(int)
        i1 = np.round((p1 - left1) / fine - counts / 2).astype(int)

        # tile each cell over its block, grouped by block size
        matrix = np.full((n1, n0), np.nan)
        for n in np.unique(counts):
            PeleLMeXPostProcessor._fill_blocks(matrix, vals, counts == n, i0, i1, n)

        # finest cell-center coordinates along each axis
        c0 = left0 + (np.arange(n0) + 0.5) * fine
        c1 = left1 + (np.arange(n1) + 0.5) * fine
        return matrix, c0, c1

    # =========================================================================
    # Dimension check with a list
    def _check_dimension(self, data_list: list, expected_dim: int) -> bool:
        """
        Check if the data list has the expected dimension.

        Parameters
        ----------
        data_list : list
            List of data values (e.g., coordinates)
        expected_dim : int
            Expected dimension (e.g., 2 for 2D, 3 for 3D)

        Returns
        -------
        bool
            True if dimension matches, False otherwise
        """
        if len(data_list) != expected_dim:
            print(f"Error: Expected {expected_dim} values, but got {len(data_list)}. Data: {data_list}")
            return False
        return True
    
    
    # =========================================================================
    # Plot slice with imshow
    def plot_df_slice(self, df: pd.DataFrame, field:str, plane:str = "xy", 
                      save_dir: str = "./figs/", cmap = "hot", time = None,
                      vmin = None, vmax = None, show = False,
                      title = None):
        """
        Plot a 2D slice using imshow.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the slice data with coordinates and field values.
        field : str
            Name of the field to plot.
        plane : str, optional
            Coordinate plane for the slice ('xy', 'xz', or 'yz'). Default is 'xy'.
        save_dir : str, optional
            Directory to save the plot. Default is './figs/'.
        cmap : str, optional
            Colormap for the plot. Default is 'hot'.
        time : float, int, or None, optional
            Time specification for the plot. Default is None, and no save in this case.
        vmin : float, optional
            Minimum value for color scaling. If None, uses the min of the field data.
        vmax : float, optional
            Maximum value for color scaling. If None, uses the max of the field data.
        show : bool, optional
            If True, displays the plot. Default is False.
        title : str, optional
            Title for the plot. Default is None.    

        Returns
        -------
        None
        """
        # initialize data
        ff, xx, yy = self.expand_df_to_matrix(df, field, plane)
        
        # calculate figsize
        axis_dict = {"xy": (0, 1), "xz": (0, 2), "yz": (2, 1)}
        left = self.domain_left
        right = self.domain_right
        axis1, axis2 = axis_dict[plane]
        xxmin, xxmax = left[axis1], right[axis1]
        yymin, yymax = left[axis2], right[axis2]
        figh = 4
        figw = 4 * (xxmax - xxmin) / (yymax - yymin) 
        
        # initialize vmin/vmax
        if vmin is None:
            vmin  = np.min(ff)
            vmax  = np.max(ff)
        
        fig, ax = plt.subplots(figsize = (figw, figh)) 

        im = ax.imshow(ff, extent = [xxmin * 1000, xxmax * 1000, yymin * 1000, yymax * 1000],
                origin = "lower", cmap = cmap, vmin = vmin, vmax = vmax)
        if (plane == "xy"):
            ax.set_xlabel("x (mm)", fontsize = 8)
            ax.set_ylabel("y (mm)", fontsize = 8)
        elif (plane == "xz"):
            ax.set_xlabel("x (mm)", fontsize = 8)
            ax.set_ylabel("z (mm)", fontsize = 8)
        elif (plane == "yz"):
            ax.set_xlabel("z (mm)", fontsize = 8)
            ax.set_ylabel("y (mm)", fontsize = 8)
        ax.set_xlabel("x (mm)", fontsize = 8)
        ax.set_ylabel("y (mm)", fontsize = 8)
        if title is not None:
            ax.set_title(title, fontsize = 8)
        fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
        # tight layout
        plt.tight_layout()
        
        # create dir if not exit
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save
        if time is not None:
            plt.savefig(f"{save_dir}/{field}_{time:.3e}.png", dpi = 300)
        # plt.colorbar()
        if not show:
            plt.close()
  

    # =========================================================================
    # Volume Rendering
    # =========================================================================

    def volume_render(self, field: str, time=None,
                      camera_position: list = None,
                      focus: list = None,
                      rotation: float = None,
                      zoom: float = 3.0,
                      value_range: tuple = None,
                      colormap: str = "inferno",
                      sigma_clip: float = 10.0,
                      name: str = None) -> None:
        """
        Generate a volume rendering of the specified field (3D only).

        Parameters
        ----------
        field : str
            Field name to render (e.g., "temp")
        time : float, int, or None
            Time specification
        camera_position : list, optional
            [x, y, z] camera position in meters.
            Use suggest_camera_settings() for suggestions.
        focus : list, optional
            [x, y, z] camera focus point in meters. Default: domain center.
        rotation : float, optional
            Camera rotation angle in radians (roll). Default: no rotation.
            Common values: np.pi/2 (90°), np.pi (180°), 3*np.pi/2 (270°)
        zoom : float
            Camera zoom factor. > 1 zooms in, < 1 zooms out. Default: 3.0
        value_range : tuple, optional
            (min, max) value range for transfer function.
            If None, uses data min/max.
        colormap : str
            Colormap name (default: "inferno")
        sigma_clip : float
            Sigma clipping value for output (default: 10.0)
        name : str
            Name for the output file (required).
            Output: {output_dir}/vol/{name}.{field}.t{time}.png
        """
        if name is None:
            raise ValueError("Please provide a 'name' for the output file")

        ds = self._resolve_timestep(time)
        current_time = float(ds.current_time)

        field_tuple = ('boxlib', field)

        # Create scene
        sc = yt.create_scene(ds, field=field_tuple)
        source = sc[0]

        # Determine value range
        if value_range is None:
            ad = ds.all_data()
            mi = float(ad[field_tuple].min())
            ma = float(ad[field_tuple].max())
        else:
            mi, ma = value_range

        # Create transfer function
        tf = yt.ColorTransferFunction((mi, ma))
        tf.add_layers(5, colormap=colormap, w=0.02)
        source.tf = tf

        # Set up camera
        cam = sc.camera

        # Camera position
        if camera_position is not None:
            cam.position = ds.arr(camera_position, 'm')
        else:
            # Default: view from negative x direction
            center = ds.domain_center.to('m').value
            width = ds.domain_width.to('m').value
            cam.position = ds.arr([center[0] - 3*width[0], center[1], center[2]], 'm')

        # Camera focus
        if focus is not None:
            cam.focus = ds.arr(focus, 'm')
        else:
            cam.focus = ds.domain_center

        # Camera rotation (roll)
        if rotation is not None:
            cam.roll(rotation)

        # Zoom
        cam.zoom(zoom)

        # Generate output path
        output_path = self._generate_output_path('vol', name, field, current_time)

        sc.save(output_path, sigma_clip=sigma_clip)
        print(f"Volume rendering saved to: {output_path}")

    def volume_render_all_times(self, field: str, name: str,
                                time_range: tuple = None,
                                camera_position: list = None,
                                focus: list = None,
                                rotation: float = None,
                                zoom: float = 3.0,
                                value_range: tuple = None,
                                colormap: str = "inferno",
                                sigma_clip: float = 10.0) -> None:
        """
        Generate volume renderings for all timesteps (3D only).

        Parameters
        ----------
        field : str
            Field name to render
        name : str
            Name for output files
        time_range : tuple, optional
            (start_time, end_time) to filter timesteps
        camera_position : list, optional
            [x, y, z] camera position in meters
        focus : list, optional
            [x, y, z] camera focus point in meters
        rotation : float, optional
            Camera rotation angle in radians
        zoom : float
            Camera zoom factor
        value_range : tuple, optional
            Value range for transfer function
        colormap : str
            Colormap name
        sigma_clip : float
            Sigma clipping value
        """
        filtered = self._filter_times(time_range)

        for idx, t in filtered:
            self.volume_render(field, time=idx,
                              camera_position=camera_position,
                              focus=focus,
                              rotation=rotation,
                              zoom=zoom,
                              value_range=value_range,
                              colormap=colormap,
                              sigma_clip=sigma_clip,
                              name=name)

    @staticmethod
    def create_gif(field, dir = "./figs", fps = 3):
        files = os.listdir(dir)
        target_files = [f for f in files if f.startswith(field) and f.endswith(".png")]
        target_files.sort(key=lambda x: float(x.replace(".png","").replace(f"{field}_","")))
        imgs = [Image.open(f"{dir}/{f}") for f in target_files]
        imgs[0].save(f"{dir}/{field}.gif", save_all = True, append_images = imgs[1:], duration = 1000//fps, loop = 0)
