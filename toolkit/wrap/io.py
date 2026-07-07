import pandas as pd
from pandas import DataFrame
import numpy as np
from os.path import join
from toolkit.wrap.wrapdata import WRAP_IDS, COL_FLOW_RIGHTS, COL_CONTROL_POINTS, COL_DIVERSION_RIGHTS, COL_RESERVOIR, N_HEADER, COL_WATER_RIGHT, COL_CONTROL_POINT
from pathlib import Path
       
    
def df_to_evp(evap_df: DataFrame, file_name: str):
    """Converts a dataframe with evap information to WRAP compatible .EVA file

    Parameters
    ----------
    evap_df : DataFrame
        pandas dataframe with a year column and month column represented as integers.
    file_name : str
        path to .EVA file to be written
    """
    with open(file_name, "wt") as file:
        years = evap_df.index.year.unique()
        sites = [site for site in evap_df.columns if "EV" in site]
        for year in years:
            year_df = evap_df[evap_df.index.year == year]
            for site in sites:
                line = f"{site}{year:>8}"
                for num in year_df[site]:
                    num = float(num)
                    line += f"{num: 8.3f}"
                line += "\n"
                file.write(line)


def df_to_flo(flo_df: DataFrame, filename: str):
    """credit to Stephen/Travis

    Parameters
    ----------
    flo_df : DataFrame
        _description_
    filename : str
        _description_
    """
    # IDs = pd.Series(WRAP_IDS)
    IDs = flo_df.columns.to_list()
    stations = flo_df.shape[1]

    # Years of Monthly Data
    start_year = flo_df.index.min().year
    num_years = int(flo_df.shape[0] / 12)

    # Create dataframe to populate with formatted FLO data
    formatted_data = pd.DataFrame(data=np.zeros([stations * num_years, 14]), dtype=object)

    # Format Node ID and Year columns
    Years = list(flo_df.index.year.unique())
    Years_repeating = list(flo_df.index.year)
    CP_col = pd.Series(index=range(num_years * stations), dtype="string")
    years_FLO = np.zeros(num_years * stations)
    for i in range(num_years):
        years_FLO[i * stations : (i + 1) * stations] = np.ones(stations) * Years[i]
        CP_col.iloc[i * stations : (i + 1) * stations] = IDs

    formatted_data[0] = CP_col.astype("str")
    formatted_data[1] = years_FLO.astype("int")

    for i in range(12):
        formatted_data.iloc[:, 2 + i] = np.zeros(stations * num_years).astype("int")

    for i in range(num_years):
        for j in range(stations):
            formatted_data.iloc[i * stations + j, 2:14] = (
                flo_df.iloc[i * 12 : (i + 1) * 12, j]
            ).astype("int")

    formatted_data = formatted_data.astype(
        {k: int for k in list(formatted_data.columns) if k != 0}
    )
    lines = []
    for line in range(num_years * stations):
        line = formatted_data.iloc[line, :].astype("str")
        formatted_line = []
        for i in range(14):
            if i == 0:
                formatted_line.append(line[0])
            else:
                padded_entry = line[i].rjust(8)
                formatted_line.append(padded_entry)

        joined_line = [
            formatted_line[0]
            + formatted_line[1]
            + formatted_line[2]
            + formatted_line[3]
            + formatted_line[4]
            + formatted_line[5]
            + formatted_line[6]
            + formatted_line[7]
            + formatted_line[8]
            + formatted_line[9]
            + formatted_line[10]
            + formatted_line[11]
            + formatted_line[12]
            + formatted_line[13]
        ]

        lines.append(joined_line)

    # saves .FLO file with streamflow realization for WRAP input #
    with open(filename, "w") as f:
        for i in range(num_years * stations):
            f.write(str(lines[i][0]))
            f.write("\n")

    return


def _monthly_wide_to_df(filename: str, csv_name: str = None):
    """Reads a WRAP "wide monthly" file (8-char site id, year, 12 monthly values per
    line) into a pandas DataFrame indexed by month, one column per site.

    Shared by `flo_to_df`, `evp_to_df`, `fad_to_df`, and `his_to_df` — all four WRAP
    input files (.FLO, .EVA, .FAD, .HIS) use this identical layout. The site id is
    always exactly 8 characters and must be read with a fixed-width slice rather than
    `line.split()`: some basins (e.g. Trinity, Sabine) right-pad short ids with an
    internal space after the 2-character record prefix (`"EV EV409"`), which
    `line.split()` would incorrectly split into two tokens.

    Parameters
    ----------
    filename : str
        path to the .FLO / .EVA / .FAD / .HIS file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Monthly data, one column per site id, indexed by a monthly DatetimeIndex.
    """
    with open(filename, "rt") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        if line[0] != "*":
            line_data = [line[:8]]  # site id, sometimes contains an internal space
            # rstrip("*"): C3.HIS has a stray "**" comment marker glued directly
            # onto its last data value with no separating whitespace ("...0**"),
            # which would otherwise fail the float cast below.
            line_data.extend(tok.rstrip("*") or "nan" for tok in line[8:].split())
            data.append(line_data)
    df = pd.DataFrame(data)
    df = df.dropna()
    df = df.pivot(index=0, columns=1).transpose().swaplevel().sort_index()
    df = df.reset_index()
    df = df.rename(columns={1: "year", "level_1": "month"})
    df["month"] = df["month"] - 1
    df["year"] = df.year.astype(int)
    df.insert(
        0,
        "date",
        df.apply(lambda row: np.datetime64(f"{row.year}-{row.month:02d}"), axis=1),
    )
    df = df.drop(columns=["year", "month"])
    df = df.set_index("date")
    df = df.astype(float)

    if csv_name:
        df.to_csv(csv_name)

    return df


def evp_to_df(file_name: str, csv_name: str = None):
    """Reads evap file into a pandas dataframe and optionally writes the data as a csv

    Parameters
    ----------
    file_name : str
        path to .EVA file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Evaporation data
    """
    return _monthly_wide_to_df(file_name, csv_name)


def flo_to_df(filename: str, csv_name: str = None):
    """Reads a natural streamflow (.FLO) file into a pandas dataframe and optionally
    writes the data as a csv

    Parameters
    ----------
    filename : str
        path to .FLO file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Streamflow data
    """
    return _monthly_wide_to_df(filename, csv_name)


def fad_to_df(filename: str, csv_name: str = None):
    """Reads a flow-at-diversion targets (.FAD) file into a pandas dataframe and
    optionally writes the data as a csv. Same site-id/year/12-monthly-value layout
    as `.FLO`.

    Parameters
    ----------
    filename : str
        path to .FAD file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Flow-at-diversion target data
    """
    return _monthly_wide_to_df(filename, csv_name)


def his_to_df(filename: str, csv_name: str = None):
    """Reads a historical reservoir operating tier (.HIS) file into a pandas
    dataframe and optionally writes the data as a csv. Same site-id/year/12-monthly-
    value layout as `.FLO`; values are small integer tier codes rather than
    continuous quantities.

    Parameters
    ----------
    filename : str
        path to .HIS file
    csv_name : str, optional
        path to csv file to write to, by default None

    Returns
    -------
    DataFrame
        Historical operating tier data
    """
    return _monthly_wide_to_df(filename, csv_name)


def _parse_fixed_width_records(lines, schema, prefix=None):
    """Parses lines against a fixed-width column schema into a DataFrame.

    Shared by `dat_to_df` (WR records, `COL_WATER_RIGHT`) and `cp_to_df` (CP
    records, `COL_CONTROL_POINT`) — both schemas follow the same convention: a
    blank field becomes `0` (int columns) or `NaN` (float columns), and a field
    that is entirely `*` characters (WRAP's overflow marker) becomes `NaN` with a
    printed warning.

    Parameters
    ----------
    lines : Iterable[str]
        Lines to parse.
    schema : list[dict]
        Column schema: dicts with 'name', 'dtype', 'length' keys, in byte order.
    prefix : str, optional
        If given, lines not starting with this prefix are skipped. If None,
        every line in `lines` is parsed.

    Returns
    -------
    DataFrame
        One row per matching line, one column per schema entry.
    """
    records = []
    for line in lines:
        if prefix is not None and not line.startswith(prefix):
            continue

        spot = 0
        datum = {}
        for col in schema:
            try:
                value = line[spot:spot + col['length']].strip()
                if (len(value) > 0) and (value == len(value) * '*'):
                    value = col['dtype'](np.nan)
                    print(f"WARNING: Value overflow for column {col['name']}:")
                    print(f"    {line}")
                elif (len(value) == 0):
                    if (col['dtype'] == np.int16):
                        value = 0
                    elif (col['dtype'] == np.float32):
                        value = col['dtype'](np.nan)
                    else:
                        value = col['dtype'](value)
                else:
                    value = col['dtype'](value)
                datum[col['name']] = value
                spot += col['length']
            except Exception as e:
                print("")
                print(f"Error on line {line}")
                print(f"Column {col['name']}")
                print("")
                raise(e)
        records.append(datum)

    return pd.DataFrame(records)


def dat_to_df(wrap_file_path, csv_name: str = None):
    """TRAVIS THURBER
    Converts a WRAP input file (.DAT extension) to a CSV file containing water rights records.
    Other records are currently ignored.

    :param wrap_file_path: path to the WRAP .DAT file

    :return: None
    """
    file_path = Path(wrap_file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    water_rights = _parse_fixed_width_records(lines, COL_WATER_RIGHT, prefix='WR')

    # creat csv file
    if csv_name:
        water_rights.to_csv(csv_name, index=False)
    return water_rights


def cp_to_df(wrap_file_path, csv_name: str = None):
    """Parses CP (control point) records from a WRAP .dat file.

    Each CP record defines one control point's downstream routing target and, for
    channel-loss purposes, a secondary reference control point and loss factor. See
    `kirklocal/wam_formats.md` for the byte-position derivation of `COL_CONTROL_POINT`.

    :param wrap_file_path: path to the WRAP .dat file

    :return: DataFrame with one row per CP record
    """
    file_path = Path(wrap_file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()

    control_points = _parse_fixed_width_records(lines, COL_CONTROL_POINT, prefix="CP")
    control_points = control_points.drop(columns=["reserved_1", "reserved_2"])

    if csv_name:
        control_points.to_csv(csv_name, index=False)
    return control_points


def _fixed_width_floats(line, start, width=8):
    """Slices `line[start:]` into fixed `width`-char chunks and parses each as a
    float, skipping any trailing chunk left blank by stripped trailing whitespace.
    """
    line = line.rstrip("\n\r")
    values = []
    for i in range(start, len(line), width):
        chunk = line[i:i + width].strip()
        if chunk:
            values.append(float(chunk))
    return values


def sv_to_df(wrap_file_path, csv_name: str = None):
    """Parses paired SV/SA reservoir breakpoint-table records from a WRAP .dat file.

    Every SV record (storage volume breakpoints, AF) is immediately followed by one
    SA record (surface area at each breakpoint, acres) with no id of its own. Returns
    a long-format DataFrame with one row per (reservoir, breakpoint).

    :param wrap_file_path: path to the WRAP .dat file

    :return: DataFrame with columns reservoir_name, breakpoint_index, storage_af, surface_area_ac
    """
    file_path = Path(wrap_file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()

    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("SV"):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("SA"):
                raise ValueError(f"SV record without a following SA record at line {i}: {line!r}")
            reservoir_name = line[2:8].strip()
            # Fixed-width 8-char fields, not whitespace-split: two adjacent wide
            # values (e.g. "2190" followed by "2922.157") can abut with no space
            # between them when the first value fills its full 8-char field width,
            # which `.split()` would incorrectly merge into one token.
            storage_values = _fixed_width_floats(line, 8)
            area_values = _fixed_width_floats(lines[i + 1], 8)
            for idx, (storage_af, surface_area_ac) in enumerate(zip(storage_values, area_values)):
                rows.append({
                    "reservoir_name": reservoir_name,
                    "breakpoint_index": idx,
                    "storage_af": storage_af,
                    "surface_area_ac": surface_area_ac,
                })
            i += 2
        else:
            i += 1

    storage_areas = pd.DataFrame(rows)
    if csv_name:
        storage_areas.to_csv(csv_name, index=False)
    return storage_areas


def dis_to_df(wrap_file_path):
    """Parses a WRAP .DIS file into FD (diversion system identifier) and WP (water
    right priority) records.

    `FD.fd_id` and `WP.wp_id` share a namespace with `WR.control_point_identifier`
    from the .dat file (see `kirklocal/wam_formats.md`). FD's optional prorate-CP
    list and WP's value list are both variable-length, so they are stored as list
    columns rather than expanded into fixed columns.

    :param wrap_file_path: path to the WRAP .DIS file

    :return: dict with keys "FD" and "WP", each a DataFrame
    """
    file_path = Path(wrap_file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()

    fd_rows = []
    wp_rows = []
    for line in lines:
        if line.startswith("FD"):
            fd_rows.append({
                "fd_id": line[2:8].strip(),
                "cp_id": line[10:16].strip(),
                "prorate_count": int(line[16:24].strip() or 0),
                "prorate_cps": line[24:].split(),
            })
        elif line.startswith("WP"):
            wp_rows.append({
                "wp_id": line[2:8].strip(),
                "values": [float(v) for v in line[8:].split()],
            })

    return {
        "FD": pd.DataFrame(fd_rows),
        "WP": pd.DataFrame(wp_rows),
    }


# @profile
def out_to_csvs(out_file, csv_folder, csvs_to_write=None):
    """TRAVIS THURBER
    Converts a WRAP output file (.OUT extension) to four CSV or parquet files, one for each type of data

    :param wrap_file_path: path to the WRAP .OUT file

    :return: None
    """
    csv_types = ["diversions", "flow_rights", "control_points", "reservoirs"]
    if csvs_to_write is None:
        csvs_to_write = csv_types
            
    if "diversions" in csvs_to_write:
        write_diversions = True
    else:
        write_diversions = False

    if "flow_rights" in csvs_to_write:
        write_flow_rights = True
    else:
        write_flow_rights = False
        
    if "control_points" in csvs_to_write:
        write_control_points = True
    else:
        write_control_points = False
        
    if "reservoirs" in csvs_to_write:
        write_reservoirs = True
    else:
        write_reservoirs = False

            

    # read the lines from the file
    file_path = Path(out_file)
    print(f"Parsing WRAP file {file_path.name}...")
    f = open(file_path, 'r')
    lines = f.readlines()

    # read and parse the meta data line
    meta = lines[N_HEADER - 1].split()
    start_year = int(meta[0])
    n_years = int(meta[1])
    n_water_rights = int(meta[3])
    n_control_points = int(meta[2])
    n_reservoirs = int(meta[4])

    # create lists for storing each type of data
    data_diversions = []
    data_flow_rights = []
    data_control_points = []
    data_reservoirs = []

    # loop through each year and month of data
    for i_year in np.arange(n_years):
        for i_month in np.arange(12):
            n_month = i_year * 12 + i_month

            # loop through each line of diversion/flow right data, and split into column
            for line in lines[
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) :
                N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights
            ]:

                # current position in line
                spot = 0

                # dictionary for the data in this line
                datum = {}

                # determine if this line is a flow right or a diversion
                is_flow_right = line.startswith('IF')
                # add the year if flow right
                if is_flow_right:
                    datum['year'] = np.int16(start_year + i_year)

                # loop through each column and parse diversion or flow right data from the line
                for col in (COL_FLOW_RIGHTS if is_flow_right else COL_DIVERSION_RIGHTS):
                    if col['name'] != 'IF':
                        value = line[spot:spot + col['length']].strip()
                        if (len(value) > 0) and (value == len(value) * '*'):
                            value = col['dtype'](np.nan)
                            # print(f"WARNING: Value overflow for {'flow_right' if is_flow_right else 'diversion'} for column {col['name']} in year {start_year + i_year} month {i_month + 1}:")
                            # print(f"    {line}")
                        else:
                            value = col['dtype'](value)
                        datum[col['name']] = value
                    spot += col['length']
                if is_flow_right:
                    data_flow_rights.append(datum)
                else:
                    data_diversions.append(datum)

            # loop through each line of control point data, and split into column
            if write_control_points:
                for line in lines[
                    N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights :
                    N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights + n_control_points
                ]:

                    # current position in line
                    spot = 0

                    # dictionary for the data in this line
                    datum = {}

                    # add the year and month
                    datum['year'] = np.int16(start_year + i_year)
                    datum['month'] = np.int16(i_month + 1)

                    # loop through each column and parse control point data from the line
                    for col in COL_CONTROL_POINTS:
                        value = line[spot:spot + col['length']].strip()
                        if (len(value) > 0) and (value == len(value) * '*'):
                            value = col['dtype'](np.nan)
                            # print(f"WARNING: Value overflow for control_point for column {col['name']} in year {start_year + i_year} month {i_month + 1}:")
                            # print(f"    {line}")
                        else:
                            value = col['dtype'](value)
                        datum[col['name']] = value
                        spot += col['length']
                    data_control_points.append(datum)

            # loop through each line of reservoir data, and split into column
            if write_reservoirs:
                for line in lines[
                    N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights + n_control_points :
                    N_HEADER + (n_month * (n_water_rights + n_control_points + n_reservoirs)) + n_water_rights + n_control_points + n_reservoirs
                ]:

                    # current position in line
                    spot = 0

                    # dictionary for the data in this line
                    datum = {}

                    # add the year and month
                    datum['year'] = np.int16(start_year + i_year)
                    datum['month'] = np.int16(i_month + 1)

                    # loop through each column and parse reservoir data from the line
                    for col in COL_RESERVOIR:
                        value = line[spot:spot + col['length']].strip()
                        if (len(value) > 0) and (value == len(value) * '*'):
                            value = col['dtype'](np.nan)
                            # print(f"WARNING: Value overflow for reservoir for column {col['name']} in year {start_year + i_year} month {i_month + 1}:")
                            # print(f"    {line}")
                        else:
                            value = col['dtype'](value)
                        datum[col['name']] = value
                        spot += col['length']
                    data_reservoirs.append(datum)

    # create data frames from each type of data
    if write_diversions:
        data_diversions = pd.DataFrame(data_diversions)
        data_diversions.to_csv(join(csv_folder, f'{file_path.stem}_diversions.csv'), index=False)
        print(f'{file_path.stem}_diversions.csv')
    
    if write_flow_rights:
        data_flow_rights = pd.DataFrame(data_flow_rights)
        data_flow_rights.to_csv(join(csv_folder, f'{file_path.stem}_flow_rights.csv'), index=False)
        print(f'{file_path.stem}_flow_rights.csv')
    
    if write_control_points:
        data_control_points = pd.DataFrame(data_control_points)
        data_control_points.to_csv(join(csv_folder, f'{file_path.stem}_control_points.csv'), index=False)
        print(f'{file_path.stem}_control_points.csv')
    
    if write_reservoirs:
        data_reservoirs = pd.DataFrame(data_reservoirs)
        data_reservoirs.to_csv(join(csv_folder, f'{file_path.stem}_reservoirs.csv'), index=False)
        print(f'{file_path.stem}_reservoirs.csv')


def out_to_dfs(out_file, dfs_to_parse=None):
    """Parse a WRAP .OUT file directly to DataFrames without writing to disk.

    Parameters
    ----------
    out_file : str or Path
        Path to the WRAP .OUT file.
    dfs_to_parse : list[str], optional
        Subset of {"diversions", "flow_rights", "control_points", "reservoirs"}.
        Defaults to ["diversions", "reservoirs"].

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by the names in dfs_to_parse.
    """
    if dfs_to_parse is None:
        dfs_to_parse = ["diversions", "reservoirs"]

    parse_cp = "control_points" in dfs_to_parse
    parse_res = "reservoirs" in dfs_to_parse

    file_path = Path(out_file)
    with open(file_path, "r") as f:
        lines = f.readlines()

    meta = lines[N_HEADER - 1].split()
    start_year = int(meta[0])
    n_years = int(meta[1])
    n_control_points = int(meta[2])
    n_water_rights = int(meta[3])
    n_reservoirs = int(meta[4])

    data_diversions = []
    data_flow_rights = []
    data_control_points = []
    data_reservoirs = []

    block = n_water_rights + n_control_points + n_reservoirs

    for i_year in np.arange(n_years):
        for i_month in np.arange(12):
            n_month = i_year * 12 + i_month
            base = N_HEADER + n_month * block

            for line in lines[base : base + n_water_rights]:
                spot = 0
                datum = {}
                is_flow_right = line.startswith("IF")
                if is_flow_right:
                    datum["year"] = np.int16(start_year + i_year)
                for col in (COL_FLOW_RIGHTS if is_flow_right else COL_DIVERSION_RIGHTS):
                    if col["name"] != "IF":
                        value = line[spot : spot + col["length"]].strip()
                        if len(value) > 0 and value == len(value) * "*":
                            value = col["dtype"](np.nan)
                        else:
                            value = col["dtype"](value)
                        datum[col["name"]] = value
                    spot += col["length"]
                (data_flow_rights if is_flow_right else data_diversions).append(datum)

            if parse_cp:
                for line in lines[base + n_water_rights : base + n_water_rights + n_control_points]:
                    spot = 0
                    datum = {"year": np.int16(start_year + i_year), "month": np.int16(i_month + 1)}
                    for col in COL_CONTROL_POINTS:
                        value = line[spot : spot + col["length"]].strip()
                        datum[col["name"]] = col["dtype"](np.nan) if (len(value) > 0 and value == len(value) * "*") else col["dtype"](value)
                        spot += col["length"]
                    data_control_points.append(datum)

            if parse_res:
                for line in lines[base + n_water_rights + n_control_points : base + block]:
                    spot = 0
                    datum = {"year": np.int16(start_year + i_year), "month": np.int16(i_month + 1)}
                    for col in COL_RESERVOIR:
                        value = line[spot : spot + col["length"]].strip()
                        datum[col["name"]] = col["dtype"](np.nan) if (len(value) > 0 and value == len(value) * "*") else col["dtype"](value)
                        spot += col["length"]
                    data_reservoirs.append(datum)

    result = {}
    if "diversions" in dfs_to_parse:
        result["diversions"] = pd.DataFrame(data_diversions)
    if "flow_rights" in dfs_to_parse:
        result["flow_rights"] = pd.DataFrame(data_flow_rights)
    if parse_cp:
        result["control_points"] = pd.DataFrame(data_control_points)
    if parse_res:
        result["reservoirs"] = pd.DataFrame(data_reservoirs)
    return result


def process_right_sectors(dat_file_path, filter_sectors=True, sectors=None):
    dat = pd.read_csv(dat_file_path)
    if filter_sectors:
        if sectors is None:
            sectors = ["IND", "IRR", "MIN", "MUN", "POW", "REC"]

        def process_use(row):
            for sector in sectors:
                try:
                    if sector in row.use:
                        return sector
                except TypeError:
                    return "nan"

        dat.use = dat.apply(process_use, axis=1)
        dat = dat[dat.use.isin(sectors)]
    
    return dat