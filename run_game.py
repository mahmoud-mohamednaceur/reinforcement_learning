'''

The main purpose of this code is to develop, train, and evaluate a reinforcement learning agent in a custom "Snake" game environment. The game runs multiple training rounds, allowing the agent to learn through trial and error by accumulating experience over thousands of episodes. The following key objectives drive the code:

Agent Training: The agent uses reinforcement learning to optimize its decision-making. Through interaction with the game environment, it learns to maximize the score and minimize "deaths" (game-over events) by adjusting its actions based on feedback from the game.

Performance Tracking: The code logs various metrics, such as the agent's score per round, cumulative score, mean score, and time spent before each "death." These metrics allow for monitoring the agent's progress over time and fine-tuning its behavior.

Model Saving: When the agent achieves a new "best performance" score, the model is saved. This lets developers track and keep the highest-performing model configurations for later analysis or application.

Data Collection and Analysis: Key metrics are stored in a structured format and saved as a CSV file. This data enables post-training analysis and visualization, which helps evaluate the agent’s performance trends and the effectiveness of different training parameters.

Visualization: The code includes real-time plotting to visualize the agent’s performance, providing insights into its learning progression.

By combining these elements, the code aims to build an intelligent agent capable of mastering the Snake game and to generate data for evaluating and improving the model's learning process.


'''


import pygame
import os
import pandas  as  pd
from game_lib.snake_game_functions import  *

OUTPUT_PATH = "analysis_output_folder/analysis_csv"
CSV_FILE = os.path.join(OUTPUT_PATH, "agent_statistics_over_training_epochen.csv")

if __name__ == '__main__':
    # Initialize game and agent
    game = SnakeGame()
    agent = Create_agent(11, 256, 512, 3, 0.001, 1000, 100_000, 0.99)

    # Initialize metrics and counters
    step, best_performance, total_score, total_time = 0, 0, 0, 0
    score_list, total_score_list, mean_score_list = [], [], []
    data = {
        'n_games': [], 'playerScoreProRound': [], 'playerTotalScore': [],
        'PlayedTimeBeforeDeath': [], 'TotalPlayedTimeBeforeDeath': [],
        'MeanScore': [], 'DeviationFromScore': [], 'TimeOverScore': [],
        'TotalNumberofDeath': [], 'TimeOverDeath': [], 'Epsilon': []
    }

    for i in range(10000):
        # Get state and make a move
        state_old = game.get_state()
        final_move = agent.choose_action(state_old)

        # Perform move and update game state
        reward, done, score, time_played = game.play_step(final_move)
        state_new = game.get_state()

        agent.short_mem(state_old, state_new, final_move, reward, done)
        agent.mem.store_action(state_old, state_new, final_move, reward, done)

        # If game over, update metrics
        if done:
            agent.n_games += 1
            step += 1
            total_score += score
            total_time += time_played

            # Update best performance and save model if improved
            if score > best_performance:
                best_performance = score
                agent.save()

            # Calculate and store performance metrics
            mean_score = total_score / agent.n_games
            time_over_score = total_time / total_score if total_score else total_time
            time_over_death = total_time / step if step else total_time
            variance = np.var(np.array(total_score_list))

            # Append metrics to lists
            score_list.append(score)
            total_score_list.append(total_score)
            mean_score_list.append(mean_score)

            # Store metriics in data dictionary
            data['n_games'].append(step)
            data['playerScoreProRound'].append(score)
            data['playerTotalScore'].append(total_score)
            data['PlayedTimeBeforeDeath'].append(time_played)
            data['TotalPlayedTimeBeforeDeath'].append(total_time)
            data['MeanScore'].append(mean_score)
            data['DeviationFromScore'].append(variance)
            data['TimeOverScore'].append(time_over_score)
            data['TotalNumberofDeath'].append(step)
            data['TimeOverDeath'].append(time_over_death)
            data['Epsilon'].append(agent.epsilon)

            # Plot the current score and mean score
            plot(score_list, mean_score_list)

            # Reset game and update long-term memory
            game.reset()
            agent.long_mem()

    # Quit game and save   data
    pygame.quit()
    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False, mode='a', header=not os.path.exists(CSV_FILE))
