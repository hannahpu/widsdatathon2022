"""
data_process_utils.py
"""
import sys
from typing import List

import numpy as np
import pandas as pd

def backfill_energy_star_rating(
        input_df:pd.DataFrame,
        groupby_list:List[str],
        energy_star_rating_colname: str = "energy_star_rating" ,
        agg_approach_func: callable = np.nanmean
):
    """
    Backfill energy_star_rating by taking a specific aggregation approach
    (default as np.nanmean) on a grouped combinations (
    such as 'year_factor' - 'state_factor' - 'year_built' combinations )

    Parameters
    ----------
    input_df:pd.DataFrame
        The input dataframe with energy_star_rating_colname column
    groupby_list:List[str]
        A list of combinations to aggregate, e.g.
        ['year_factor', 'state_factor', 'year_built']
    energy_star_rating_colname: str
        Default as "energy_star_rating"
    agg_approach_func: callable
        Default as mean using np.nanmean
    Returns
    -------

    """
    # Get a mapping between energy star with a specific combination
    # of unique identifiers
    mapping_energy_star_per_combo_df = input_df.groupby(groupby_list).agg(
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

if __name__ == '__main__':

    wids_path = "/content/widsdatathon2022"
    sys.path.append(wids_path)

    train_df = pd.read_csv(f"{wids_path}/data/train.csv")
    train_df.columns = train_df.columns.str.lower()

    backfilled_energy_star_train_df = backfill_energy_star_rating(
        input_df=train_df.query("year_built !=0"),
        groupby_list=['year_factor', 'state_factor', 'year_built'],
        energy_star_rating_colname="energy_star_rating",
        agg_approach_func=np.nanmean
    )

    # For training, use this approach increases valid energy star rating
    # from 49042(65%) to 74058(98%)
    print(backfilled_energy_star_train_df.filter(like="energy").info())
