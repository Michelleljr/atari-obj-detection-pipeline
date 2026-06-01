import jax
import jaxatari

#pong environment
env = jaxatari.make("pong")

#Jax eandom keys needed to start
key = jax.random.PRNGKey(42)

#load first frame of game
obs, state = env.reset(key)

print("Environment successfully loaded!")
print("--- Initial Game State ---")
print(f"Player Y Position: {state.player_y}")
print(f"Enemy Y Position:  {state.enemy_y}")
print(f"Ball Coordinates:  (X: {state.ball_x}, Y: {state.ball_y})")
print(f"Current Score:     Player {state.player_score} - {state.enemy_score} Enemy")