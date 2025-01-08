from flask import Flask, request, jsonify
from stable_baselines3 import PPO
import QuizEnv  # Import your custom environment

app = Flask(__name__)

# Load the RL model and environment
model = PPO.load("quiz_rl_model.zip")
quiz_env = QuizEnv.QuizEnv("/Users/kavib/Desktop/researche/piyumila/omesh.xlsx")


@app.route('/get_question', methods=['GET'])
def get_question():
    """Retrieve the current question."""
    current_question = quiz_env.current_question
    if current_question is None:
        return jsonify({"error": "No more questions available!"}), 400
    return jsonify(current_question.to_dict())  # Convert the question to JSON


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Submit an answer to the current question."""
    try:
        data = request.json
        action = int(data['action'])  # User-selected action (index of the option)
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid or missing 'action' field"}), 400

    # Take a step in the environment
    state, reward, done, _ = quiz_env.step(action)
    return jsonify({
        "state": state,
        "reward": reward,
        "done": done,
    })


if __name__ == '__main__':
    app.run(debug=True)
