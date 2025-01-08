from flask import Flask, request, jsonify
from stable_baselines3 import PPO
import gym
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import os
import random
 
# Initialize Flask app and database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quiz_game.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Create an application context
app_context = app.app_context()
app_context.push()



with app.app_context():
    db.create_all()

try:
    custom_objects = {
        "learning_rate": 0.0003,
        "clip_range": 0.2,
        "n_steps": 2048
    }
    model = PPO.load("quiz_rl_model.zip", custom_objects=custom_objects)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class QuizEnv(gym.Env):
    def __init__(self, question_file: str):
        super(QuizEnv, self).__init__()
        try:
            self.questions_df = pd.read_excel(question_file)
            print("Available columns:", self.questions_df.columns.tolist())  # Debug print
            
            # Validate required columns exist
            required_columns = ['Level', 'Question', 'Option 1', 'Option 2', 'Option 3', 'Option 4', 'Answer']
            missing_columns = [col for col in required_columns if col not in self.questions_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
        except Exception as e:
            print(f"Error loading questions file: {e}")
            raise

        self.levels = ['Beginner', 'Intermediate', 'Expert']
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Dict({
            "current_level": gym.spaces.Discrete(len(self.levels)),
            "consecutive_correct": gym.spaces.Discrete(10),
            "consecutive_wrong": gym.spaces.Discrete(10),
        })
        self.reset()

    def reset(self):
        self.current_level = 'Beginner'
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.current_question = self._get_question()
        return self._get_state()

    def _get_valid_options(self, question):
        """Get non-null options from the question."""
        options = []
        for i in range(1, 6):  # Check options 1-5
            option_key = f'Option {i}'
            if option_key in question and pd.notna(question[option_key]):
                options.append(question[option_key])
        return options

    def _get_question(self):
        try:
            available_questions = self.questions_df[
                self.questions_df['Level'] == self.current_level
            ]
            if available_questions.empty:
                return None
            return available_questions.sample(n=1).iloc[0]
        except Exception as e:
            print(f"Error getting question: {e}")
            return None

    def _get_state(self):
        return {
            "current_level": self.levels.index(self.current_level),
            "consecutive_correct": self.consecutive_correct,
            "consecutive_wrong": self.consecutive_wrong,
        }

    def step(self, action):
        if self.current_question is None:
            return self._get_state(), 0, True, {}

        try:
            # Get only valid options
            options = self._get_valid_options(self.current_question)
            correct_answers = str(self.current_question['Answer']).split(',')

            # Validate action against available options
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

            if self.consecutive_correct >= 3:
                current_level_index = self.levels.index(self.current_level)
                if current_level_index < len(self.levels) - 1:
                    self.current_level = self.levels[current_level_index + 1]

            done = self.consecutive_wrong >= 3 or self.current_question is None
            self.current_question = self._get_question()
            return self._get_state(), reward, done, {}
            
        except Exception as e:
            print(f"Error in step function: {e}")
            return self._get_state(), 0, True, {'error': str(e)}

@app.route('/start_game', methods=['POST'])
def start_game():
    data = request.json
    user_role = data.get("role")
    user_chapter = data.get("chapter")

    if not user_role or not user_chapter:
        return jsonify({"error": "Missing role or chapter"}), 400

    session_id = request.headers.get('X-Session-ID') or str(random.randint(1000, 9999))

    try:
        new_user = User(role=user_role, chapter=user_chapter, points=0, 
                       current_level='Beginner', session_id=session_id)
        db.session.add(new_user)
        db.session.commit()

        quiz_env = QuizEnv("omesh.xlsx")
        initial_state = quiz_env.reset()

        if model is not None:
            action, _ = model.predict(initial_state)
        else:
            action = quiz_env.action_space.sample()

        new_state, reward, done, _ = quiz_env.step(action)

        if quiz_env.current_question is None:
            return jsonify({"error": "No questions available"}), 500

        # Get only valid options
        options = quiz_env._get_valid_options(quiz_env.current_question)
        
        return jsonify({
            "session_id": session_id,
            "question": quiz_env.current_question['Question'],
            "options": options,
            "state": new_state,
            "reward": reward
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Error starting game: {str(e)}"}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        data = request.json
        user_answer = int(data.get('action'))
        session_id = request.headers.get('X-Session-ID')

        if not session_id:
            return jsonify({"error": "No session ID provided"}), 400

        user = User.query.filter_by(session_id=session_id).first()
        if not user:
            return jsonify({"error": "No active game session"}), 400

        quiz_env = QuizEnv("omesh.xlsx")
        state = quiz_env._get_state()

        if model is not None:
            action, _ = model.predict(state)
        else:
            action = quiz_env.action_space.sample()

        new_state, reward, done, _ = quiz_env.step(user_answer)

        user.points += reward
        user.current_level = quiz_env.current_level
        if done:
            user.end_status = True
        db.session.commit()

        response_data = {
            "state": new_state,
            "reward": reward,
            "done": done,
            "points": user.points
        }

        if quiz_env.current_question is not None:
            options = quiz_env._get_valid_options(quiz_env.current_question)
            response_data.update({
                "question": quiz_env.current_question['Question'],
                "options": options
            })

        return jsonify(response_data)

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Error submitting answer: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)