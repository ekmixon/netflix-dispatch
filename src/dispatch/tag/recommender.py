"""
.. module: dispatch.tag.recommender
    :platform: Unix
    :copyright: (c) 2019 by Netflix Inc., see AUTHORS for more
    :license: Apache, see LICENSE for more details.
"""
import logging
from typing import List, Any
from collections import defaultdict

import tempfile
import pandas as pd
from pandas.core.frame import DataFrame

from dispatch.database.core import SessionLocal
from dispatch.tag import service as tag_service

log = logging.getLogger(__name__)


def save_model(dataframe: DataFrame, organization_slug: str, project_slug: str, model_name: str):
    """Saves a correlation dataframe to disk."""
    file_name = f"{tempfile.gettempdir()}/{organization_slug}-{project_slug}-{model_name}.pkl"
    dataframe.to_pickle(file_name)


def load_model(organization_slug: str, project_slug: str, model_name: str):
    """Loads a correlation dataframe from disk."""
    file_name = f"{tempfile.gettempdir()}/{organization_slug}-{project_slug}-{model_name}.pkl"
    return pd.read_pickle(file_name)


def correlation(df, tag_a, tag_b):
    """Determine the probability of correlation/association."""
    # Find all rows where a AND b == True
    a_and_b = df[(df[tag_a]) & (df[tag_b])]

    # Find all rows where a == True AND b != True
    a_not_b = df[(df[tag_a]) & ~(df[tag_b])]
    # Find all rows where b == True AND a != True
    b_not_a = df[(df[tag_b]) & ~(df[tag_a])]

    # Calculate the number of positive and possible outcomes using the shape attribute
    possible_outcomes = (
        a_and_b.shape[0] + a_not_b.shape[0] + b_not_a.shape[0]
    )  # shape[0] returns the number of rows
    positive_outcomes = a_and_b.shape[0]

    return positive_outcomes / possible_outcomes


def correlate_with_every_tag(df, tag_a):
    """Create correlations between every tag."""

    unique_tags = list(df.columns)
    return [correlation(df, tag_a, tag_b) for tag_b in unique_tags]


def get_unique_tags(items: List[Any]):
    """Get unique tags."""
    unique_tags = {}
    for i in items:
        for t in i.tags:
            unique_tags[t.id] = t.id
    return unique_tags


def create_correlation_dataframe(dataframe):
    """Create the correlation dataframe based on the boolean dataframe."""
    unique_tags = list(dataframe.columns)

    correlation_matrix_dict = {
        tag_a: correlate_with_every_tag(dataframe, tag_a)
        for tag_a in unique_tags
    }


    correlated_dataframe = pd.DataFrame(correlation_matrix_dict)
    correlated_dataframe["index"] = unique_tags
    return correlated_dataframe.set_index("index")


def create_boolean_dataframe(items: List[Any]):
    """Create a boolean dataframe with tag and item data."""
    unique_tags = get_unique_tags(items)
    boolean_df = pd.DataFrame(columns=unique_tags.values())

    data_dict = defaultdict(list)
    for col in boolean_df:
        for i in items:
            tag_ids = [t.id for t in i.tags]
            data_dict[col].append(col in tag_ids)

    for col in boolean_df:
        boolean_df[col] = data_dict[col]

    return boolean_df


def find_correlations(dataframe, tag):
    """Find all correlations for the given tag."""
    # Setup empty list
    correlations = []
    columns = []

    # Loop through all column at the row with the tag as its index
    for i, corr in enumerate(dataframe.loc[tag, :]):

        # Find the column
        col = dataframe.columns[i]

        # Append the correlation to the list
        correlations.append(corr)
        columns.append(col)

    return pd.DataFrame({"tag": columns, "correlation": correlations})


def find_highest_correlations(correlated_dataframe, recommendations):
    """Find the correlations with the highest relevancy."""
    # Sort the input df
    corr_df_sorted = correlated_dataframe.sort_values(by=["correlation"], ascending=False)

    return corr_df_sorted.iloc[1 : recommendations + 1]


def get_recommendations(
    db_session: SessionLocal,
    tag_ids: List[str],
    organization_slug: str,
    project_slug: str,
    model_name: str,
    recommendations: int = 5,
):
    """Get recommendations based on current tag."""
    try:
        correlation_dataframe = load_model(organization_slug, project_slug, model_name)
    except FileNotFoundError:
        log.warning(f"No model file found. ProjectName: {project_slug} ModelName: {model_name}")
        return []

    recommendations_dataframe = pd.DataFrame()
    for tag_id in tag_ids:
        correlated_dataframe = find_correlations(correlation_dataframe, tag_id)
        recommendations_dataframe = pd.concat(
            [
                recommendations_dataframe,
                find_highest_correlations(correlated_dataframe, recommendations),
            ],
            ignore_index=True,
        )

    # convert back to tag objects
    tags = [
        tag_service.get(db_session=db_session, tag_id=int(t))
        for t in recommendations_dataframe["tag"][:recommendations]
    ]

    log.debug(
        f"Making tag recommendation. RecommendedTags: {','.join([t.name for t in tags])} ModelName: {model_name}"
    )
    return tags


def build_model(items: List[Any], organization_slug: str, project_slug: str, model_name: str):
    """Builds the correlation dataframe for items."""
    boolean_dataframe = create_boolean_dataframe(items)
    correlation_dataframe = create_correlation_dataframe(boolean_dataframe)
    save_model(correlation_dataframe, organization_slug, project_slug, model_name)
