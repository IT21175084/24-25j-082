import gym
from gym import spaces
import numpy as np
import pandas as pd
import random


class QuizEnv(gym.Env):
    def __init__(self, question_file: str):
        super(QuizEnv, self).__init__()
        self.questions_df = pd.read_excel(question_file)
        self.levels = ['Beginner', 'Intermediate', 'Expert']

        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # Max 5 options (1-5)
        self.observation_space = spaces.Dict({
            "current_level": spaces.Discrete(len(self.levels)),
            "consecutive_correct": spaces.Discrete(10),
            "consecutive_wrong": spaces.Discrete(10),
        })

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the environment."""
        self.current_level = 'Beginner'
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.current_question = self._get_question()
        return self._get_state()

    def _get_question(self):
        """Retrieve a random question for the current level."""
        available_questions = self.questions_df[
            self.questions_df['Level'] == self.current_level
        ]
        if available_questions.empty:
            return None
        return available_questions.sample(n=1).iloc[0]

    def _get_state(self):
        """Return the current environment state."""
        return {
            "current_level": self.levels.index(self.current_level),
            "consecutive_correct": self.consecutive_correct,
            "consecutive_wrong": self.consecutive_wrong,
        }

    def step(self, action):
        """Perform an action and return the new state, reward, done flag, and info."""
        if self.current_question is None:
            return self._get_state(), 0, True, {}

        options = [
            self.current_question.get('Option 1'),
            self.current_question.get('Option 2'),
            self.current_question.get('Option 3'),
            self.current_question.get('Option 4'),
            self.current_question.get('Option 5 (Expert only)')
        ]
        correct_answers = str(self.current_question['Answer']).split(',')

        # Validate action and check if correct
        user_answer = options[action] if action < len(options) else None
        is_correct = user_answer and user_answer.strip().lower() in [
            answer.strip().lower() for answer in correct_answers
        ]

        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
            reward = 1
        else:
            self.consecutive_correct = 0
            self.consecutive_wrong += 1
            reward = -1

        # Update level if criteria met
        if self.consecutive_correct >= 3:
            current_level_index = self.levels.index(self.current_level)
            if current_level_index < len(self.levels) - 1:
                self.current_level = self.levels[current_level_index + 1]

        # Check if the quiz is over
        done = self.consecutive_wrong >= 3 or self.current_question is None
        self.current_question = self._get_question()
        return self._get_state(), reward, done, {}
