import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class AggregationStrategy(ABC):

    @staticmethod
    def getAggregator(strategy):
        if strategy == "ADD":
            return AdditiveAggregator()
        elif strategy == "LMS":
            return LeastMiseryAggregator()

        return None

    @abstractmethod
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        pass


class AdditiveAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        aggregated_df = group_ratings.groupby('movieId').sum()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['movieId', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['movieId'])
        return {"ADD": recommendation_list}


class LeastMiseryAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number):
        # aggregate using least misery strategy
        aggregated_df = group_ratings.groupby('movieId').min()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['movieId', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['movieId'])
        return {"LMS": recommendation_list}
