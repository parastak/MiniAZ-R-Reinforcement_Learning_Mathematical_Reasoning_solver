"""
Improved Curriculum Controller for miniAZ-R.
Tracks reward progression, enables adaptive difficulty switching,
and supports statistics collection.
"""
import numpy as np
from collections import deque
import logging


class CurriculumController:
    def __init__(self,
                 window_size=100,  # Increased window size for stability
                 patience=10,      # Added patience mechanism
                 min_episodes_per_level=200,  # Increased minimum episodes
                 thresholds=None):  # More stringent thresholds
        if thresholds is None:
            thresholds = {"easy": 0.40, "medium": 0.60, "hard": 0.80}
        self.level = ["easy", "medium", "hard"]
        self.current_level_index = 0
        self.performance_window = deque(maxlen=window_size)
        self.min_episodes_per_level = min_episodes_per_level
        self.episodes_at_level = 0
        self.thresholds = thresholds
        self.patience = patience
        self.stable_episodes = 0
        self.promotions = []

    def get_level(self):
        return self.level[self.current_level_index]

    def record_result(self, reward):
        self.performance_window.append(reward)
        self.episodes_at_level += 1

        if self.episodes_at_level < self.min_episodes_per_level:
            return

        if len(self.performance_window) >= 50:
            avg_reward = np.mean(self.performance_window)
            current_level = self.get_level()

            if self.current_level_index == 0 and avg_reward >= self.thresholds['easy']:
                self._check_promotion('easy', 'medium')

            elif self.current_level_index == 1 and avg_reward >= self.thresholds['medium']:
                self._check_promotion('medium', 'hard')

    def _check_promotion(self, current_level, next_level):
        """New method to handle level promotions with patience"""
        if self.stable_episodes < self.patience:
            self.stable_episodes += 1
            return

        self.current_level_index += 1
        self.performance_window.clear()
        self.episodes_at_level = 0
        self.stable_episodes = 0
        logging.info(f"CURRICULUM PROGRESSION: {current_level.upper()} -> {next_level.upper()}")