"""
data_process_utils.py
"""
import sys
from typing import List

import numpy as np
import pandas as pd

def backfill_energy_star_rating(
        input_df:pd.DataFrame,
        mapping_df: pd.DataFrame,
        groupby_list:List[str],
        energy_star_rating_colname: str = "energy_star_rating" ,
        agg_approach_func: callable = np.nanmean
) -> pd.DataFrame:
    """
    Backfill energy_star_rating by taking a specific aggregation approach
    (default as np.nanmean) on a grouped combinations (
    such as 'year_factor' - 'state_factor' - 'year_built' combinations )

    Parameters
    ----------
    input_df:pd.DataFrame
        The input dataframe with energy_star_rating_colname column
    mapping_df:pd.DataFrame
        The mapping_df used to backfill energy_star_rating, for both training
        and test datasets, mapping_df = train_df.
    groupby_list:List[str]
        A list of combinations to aggregate, e.g.
        ['year_factor', 'state_factor', 'year_built']
    energy_star_rating_colname: str
        Default as "energy_star_rating"
    agg_approach_func: callable
        Default as mean using np.nanmean
    Returns
    -------
    pd.DataFrame with one additional column of backfilled_energy_star_rating

    """
    # Get a mapping between energy star with a specific combination
    # of unique identifiers
    mapping_energy_star_per_combo_df = mapping_df.groupby(groupby_list).agg(
        agg_approach_func
    ).reset_index()[groupby_list + [energy_star_rating_colname]].rename(
        columns={
        energy_star_rating_colname: "energy_star_rating_backfilled"
    })

    input_w_backfilled_energy_star_df = input_df.merge(
        mapping_energy_star_per_combo_df, how= "left",
        on=groupby_list
    )
    if input_w_backfilled_energy_star_df.shape[0] != input_df.shape[0]:
        raise ValueError("Left join is incorrect")

    # Take the original energy_star_rating if existing, use
    # "energy_star_rating_backfilled" if original energy_star_rating is missing
    input_w_backfilled_energy_star_df = input_w_backfilled_energy_star_df.assign(
        backfilled_energy_star_rating=lambda df: df.apply(
            lambda row: row[energy_star_rating_colname]
            if not np.isnan(row[energy_star_rating_colname])
            else row["energy_star_rating_backfilled"]
            ,axis=1
        )
    ).drop(columns=["energy_star_rating_backfilled"])


    return input_w_backfilled_energy_star_df

def backfill_wind_direction(
        input_df:pd.DataFrame,
        mapping_df: pd.DataFrame,
        groupby_list:List[str],
        wind_direction_colname: str = "direction_max_wind_speed" ,
        agg_approach_func: callable = np.nanmean
) -> pd.DataFrame:
    """
    Backfill wind direction by taking a specific aggregation approach
    (default as np.nanmean) on a grouped combinations (
    such as 'year_factor' - 'state_factor' combinations )

    Parameters
    ----------
    input_df:pd.DataFrame
        The input dataframe with wind_direction_colname column
    mapping_df:pd.DataFrame
        The mapping_df used to backfill wind directions, for both training
        and test datasets, mapping_df = train_df.
    groupby_list:List[str]
        A list of combinations to aggregate, e.g.
        ['year_factor', 'state_factor']
    wind_direction_colname: str
        Default as "direction_max_wind_speed"
    agg_approach_func: callable
        Default as mean using np.nanmean
    Returns
    -------
    pd.DataFrame with one additional column of f"backfilled_{wind_direction_colname}"

    """
    # Get a mapping between wind directions with a specific combination
    # of unique identifiers
    mapping_wind_direction_per_combo_df = mapping_df.groupby(groupby_list).agg(
        agg_approach_func
    ).reset_index()[groupby_list + [wind_direction_colname]].rename(
        columns={
            wind_direction_colname: f"{wind_direction_colname}_backfilled"
    })

    input_w_backfilled_wind_direction_df = input_df.merge(
        mapping_wind_direction_per_combo_df, how= "left",
        on=groupby_list
    )
    if input_w_backfilled_wind_direction_df.shape[0] != input_df.shape[0]:
        raise ValueError("Left join is incorrect")

    # Take the original wind direction if existing, use
    # f"{wind_direction_colname}_backfilled" if original wind direction is missing
    input_w_backfilled_wind_direction_df[f"backfilled_{wind_direction_colname}"]=(
        input_w_backfilled_wind_direction_df.apply(
            lambda row: row[wind_direction_colname]
            if not np.isnan(row[wind_direction_colname])
            else row[f"{wind_direction_colname}_backfilled"]
            ,axis=1
        )
    ).drop(columns=[f"{wind_direction_colname}_backfilled"])


    return input_w_backfilled_wind_direction_df

def categorize_wind_direction(wind_direction_degree:float,
                              n_bins_categorized:int=8
                              ) -> str:
  """
  Categorize wind direction from 0-360 into 4, 8 or 16 directions

  Parameters
  ----------
  wind_direction_degree:float
    The wind direction degree between 0 to 360, if it is np.nan then function
    returns np.nan
  n_bins_categorized:int=8
    The number of directions categorized. Default at 8.
  Returns
  -------
  str, one of the directions in directions_list
  """
  # List out directions based on different n_bins_categorized
  if n_bins_categorized == 16:
    directions_list=["N","NNE","NE","ENE","E","ESE", "SE", "SSE",
                     "S","SSW","SW","WSW","W","WNW", "NW","NNW"]
  elif n_bins_categorized == 8:
    directions_list=["N","NE","E","SE",
                     "S","SW","W","NW"]
  elif n_bins_categorized == 4:
    directions_list=["N","E", "S","W"]
  else:
    raise ValueError("The n_bins_categorized can only be one of 4, 8 or 16.")

  if np.isnan(wind_direction_degree):
    return np.nan
  else:
    if 0 <= wind_direction_degree <= 360:
      # Get the degree delta for each bin
      deg_in_each_bin = float(360/n_bins_categorized)
      # +0.5 so that it can be in only one direction bin
      bins_index = int((wind_direction_degree/deg_in_each_bin)+ 0.5)
      return directions_list[(bins_index % n_bins_categorized)]
    else:
      raise ValueError(f"wind direction degree must between 0 to 360")


def parse_facility_type(input_df, facility_type_colname="facility_type"):
    """

    Parse facility type into main categories

    Returns
    -------
    input_df with an additional column "facility_type_parsed"

    """
    # Get a mapping between facility type with parsed
    input_df["facility_type_parsed"] = input_df[facility_type_colname].apply(
        lambda a_facility_type: a_facility_type.split("_")[0] if (
            "2to4" not in a_facility_type and "5plus" not in a_facility_type) else
        a_facility_type.split("_")[-2] + "_" + a_facility_type.split("_")[-1]

    )

    return input_df

# Sample code of using the functions
if __name__ == '__main__':

    wids_path = "/content/widsdatathon2022"
    sys.path.append(wids_path)

    train_df = pd.read_csv(f"{wids_path}/data/train.csv")
    train_df.columns = train_df.columns.str.lower()

    # Backfill energy star rating
    backfilled_energy_star_train_df = backfill_energy_star_rating(
        input_df=train_df.query("year_built !=0"),
        mapping_df=train_df.query("year_built !=0"),
        groupby_list=['year_factor', 'state_factor', 'year_built'],
        energy_star_rating_colname="energy_star_rating",
        agg_approach_func=np.nanmean
    )

    # For training, use this approach increases valid energy star rating
    # from 49042(65%) to 74058(98%)
    print(backfilled_energy_star_train_df.filter(like="energy").info())

    # Backfill wind direction
    backfilled_wind_direction_df = backfill_wind_direction(
        input_df=backfilled_energy_star_train_df,
        mapping_df=backfilled_energy_star_train_df,
        groupby_list=['year_factor', 'state_factor'],
        wind_direction_colname="direction_max_wind_speed",
        agg_approach_func=np.nanmean
    )

    # For training, use this approach increases valid direction_max_wind_speed
    # from 34674(46%) to 70224(93%)
    print(backfilled_wind_direction_df.filter(like="direction_max").info())


    # Categorize wind speed directions
    backfilled_wind_direction_df = backfilled_wind_direction_df.assign(
        categorized_direction_max_wind_speed=lambda df: df['direction_max_wind_speed'].apply(
            lambda a_direction_value: categorize_wind_direction(
                wind_direction_degree=a_direction_value, n_bins_categorized=8)))

    print(backfilled_wind_direction_df.filter(like="direction")[[
        'direction_max_wind_speed', 'categorized_direction_max_wind_speed']].drop_duplicates())

    # Get a mapping between facility type with parsed
    train_w_parsed_facility_type_df = parse_facility_type(
        input_df=backfilled_wind_direction_df,
        facility_type_colname="facility_type")

    # Reduced to around 20 facility_types
    print(train_w_parsed_facility_type_df[[
        "facility_type_parsed", "facility_type"]].drop_duplicates())
