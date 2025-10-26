How the Q-Learning AI Works
Q-Learning Overview
Q-Learning is a reinforcement learning algorithm that enables the AI (Dragon) to learn optimal strategies through trial and error. The dragon learns which actions lead to rewards (winning) and which lead to penalties (losing).

The Learning Process
1. State Representation
The AI observes the game state through these features:

Distance to knight: "very_close", "close", "medium", "far"

Health levels: "very_high", "high", "medium", "low", "very_low" (for both dragon and knight)

Knight's actions: attacking, blocking, airborne

Fireball threats: detecting incoming blue fireballs

Self-state: attacking, blocking, grounded/airborne

2. Available Actions
The dragon can choose from 13 strategic actions:

Movement: approach_fast, approach_slow, retreat_fast, retreat_slow

Combat: attack_light, attack_heavy, counter_attack

Defense: dodge_left, dodge_right, jump_evade

Strategy: jump_attack, wait_pattern, idle

3. Q-Table Structure
python
# The Q-table stores learned values for state-action pairs
###
q_table = {
    state_1: {action_1: Q_value, action_2: Q_value, ...},
    state_2: {action_1: Q_value, action_2: Q_value, ...},
    # ... more states
}
4. Learning Algorithm
The AI updates its knowledge using the Q-learning formula:

text
Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
Where:

α (alpha): Learning rate (0.15) - how quickly new information overrides old

γ (gamma): Discount factor (0.98) - importance of future rewards

R: Immediate reward received

s: Current state, a: Action taken

s': Next state, a': Possible next actions

 Reward System
Positive Rewards (+)
Damaging knight: +15 per HP point dealt

Dodging fireballs: +10

Successful counter-attacks: +8

Finishing moves: +12 when knight is low health

Winning match: +200

Negative Rewards (-)
Taking damage: -20 per HP point lost

Poor decisions: -15 for approaching when fireballs are close

Missing opportunities: -5 for idling when close to vulnerable knight

Losing match: -200

 Advanced Features
Pattern Recognition
python
# The AI remembers recent game states
self.pattern_memory = deque(maxlen=20)

# Detects repeating patterns that lead to wins
if len(set(recent_states)) < 3:  # If states are repeating
    return self.get_pattern_based_action()
Adaptive Learning
Exploration rate: Starts at 50%, decays to 2% minimum

Learning acceleration: Learning rate increases with experience

Win pattern storage: Remembers successful state-action sequences

 Training Progression
Phase 1: Exploration (Matches 1-20)
Exploration rate: 50% → 30%

Behavior: Random actions, learning basic mechanics

Focus: Understanding rewards and penalties

Phase 2: Strategy Development (Matches 21-50)
Exploration rate: 30% → 10%

Behavior: Combining learned actions into strategies

Focus: Fireball dodging, timing attacks

Phase 3: Mastery (Matches 50+)
Exploration rate: 10% → 2%

Behavior: Executing proven winning patterns

Focus: Counter-play, prediction, adaptation

 Model Persistence
Saving Progress
python
def save_model(self):
    pickle.dump({
        'q_table': self.q_table,                    # Learned state-action values
        'exploration_rate': self.exploration_rate,  # Current exploration level
        'match_count': self.match_count,            # Training experience
        'win_patterns': list(self.win_patterns)     # Successful strategies
    })
Loading Progress
Model automatically loads from enhanced_ai_model.pkl

Training continues from saved state

Win patterns are preserved between sessions

What the AI Learns
Early Learning (First 10 matches)
Basic movement and attack timing

Understanding health and damage

Simple cause-effect relationships

Intermediate Learning (10-30 matches)
Fireball dodging patterns

Effective attack ranges

Health management strategies

Counter-attack opportunities

Advanced Learning (30+ matches)
Predicting knight movement patterns

Baiting and trapping strategies

Combo execution

Adaptive playstyle based on knight behavior

 Performance Metrics
States discovered: Number of unique game situations encountered

Win patterns: Successful strategy sequences identified

Exploration rate: Balance between trying new actions vs using known good ones

Match count: Total training experience
