# scrappy_fight_ai.py
import pygame
from pygame.locals import *
import math
import sys
import random
import os
import json
import numpy as np
from collections import deque
import pickle

# ---------- Config ----------
WIDTH, HEIGHT = 900, 400
FPS = 60
GROUND_Y = HEIGHT - 60

PLAYER_WIDTH, PLAYER_HEIGHT = 40, 60

LIGHT_DAMAGE = 8
HEAVY_DAMAGE = 18
FIRE_DAMAGE = 12

LIGHT_DURATION = 12    # frames
HEAVY_DURATION = 20
FIRE_DURATION = 30
ATTACK_COOLDOWN = 25   # Increased cooldown for knight (frames between attacks)
BLOCK_REDUCTION = 0.5  # reduce damage to 50% when blocking
INVINCIBLE_FRAMES = 12 # after a successful hit, small i-frames

# ---------- Blue Fireball Projectile ----------
class BlueFireball:
    def __init__(self, x, y, direction, speed=5):
        self.x = x
        self.y = y
        self.vx = direction * speed
        self.vy = 0
        
        self.radius = 10
        self.active = True
        self.color = (100, 150, 255)
        
    def update(self):
        self.x += self.vx
        
        # Deactivate if out of bounds
        if (self.x < -50 or self.x > WIDTH + 50):
            self.active = False
            
    def draw(self, surf):
        # Draw blue fireball with gradient
        pygame.draw.circle(surf, (150, 200, 255), (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surf, (100, 150, 255), (int(self.x), int(self.y)), self.radius - 2)
        pygame.draw.circle(surf, (50, 100, 255), (int(self.x), int(self.y)), self.radius - 4)
        
    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)

# ---------- GIF Image Loader ----------
class GIFImage:
    def __init__(self, filename, scale=1.0):
        try:
            self.image = pygame.image.load(filename).convert_alpha()
            print(f"Successfully loaded: {filename}")
            
            # Scale if needed
            if scale != 1.0:
                original_size = self.image.get_size()
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                self.image = pygame.transform.scale(self.image, new_size)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # Create a colored placeholder
            self.image = pygame.Surface((60, 80), pygame.SRCALPHA)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            pygame.draw.rect(self.image, color, (0, 0, 60, 80))
            pygame.draw.rect(self.image, (255, 255, 255), (0, 0, 60, 80), 2)
        
    def get_image(self, flip=False):
        if flip:
            return pygame.transform.flip(self.image, True, False)
        return self.image

# ---------- Utility ----------
def clamp(v, a, b): return max(a, min(b, v))

# ---------- Fighter Class (shared) ----------
class Fighter:
    def __init__(self, x, color, gif_file=None, scale=1.0):
        self.x = x
        self.y = GROUND_Y - PLAYER_HEIGHT
        self.vx = 0
        self.vy = 0
        self.color = color
        self.facing = 1  # 1 = right, -1 = left
        self.on_ground = True

        self.rect = pygame.Rect(self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.hp = 100
        self.max_hp = 100

        # action state
        self.is_blocking = False
        self.is_attacking = False
        self.attack_timer = 0
        self.attack_type = None  # "light", "heavy", or "fire"
        self.attack_cooldown = 0

        self.invincible_timer = 0

        # Adjusted movement params for better jump control
        self.speed = 3.2
        self.jump_strength = -10.0  # Reduced for better control
        self.gravity = 0.15         # Reduced gravity for slower fall

        # for preventing repeated hits by same attack
        self.attack_hit_targets = set()
        
        # GIF image
        self.gif_image = None
        if gif_file:
            self.gif_image = GIFImage(gif_file, scale)
        
    def update_rect(self):
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

    def apply_physics(self):
        # horizontal velocity dampening
        self.vx *= 0.9
        self.x += self.vx
        # gravity
        self.vy += self.gravity
        self.y += self.vy

        # ground collision
        if self.y + PLAYER_HEIGHT >= GROUND_Y:
            self.y = GROUND_Y - PLAYER_HEIGHT
            self.vy = 0
            self.on_ground = True
        else:
            self.on_ground = False

        # screen bounds
        self.x = clamp(self.x, 10, WIDTH - PLAYER_WIDTH - 10)
        self.update_rect()

    def start_attack(self, atk_type):
        if self.attack_cooldown > 0 or self.is_attacking:
            return False
        self.is_attacking = True
        self.attack_type = atk_type
        
        if atk_type == "light":
            self.attack_timer = LIGHT_DURATION
            self.attack_cooldown = ATTACK_COOLDOWN
        elif atk_type == "heavy":
            self.attack_timer = HEAVY_DURATION
            self.attack_cooldown = ATTACK_COOLDOWN + 10
            
        self.attack_hit_targets.clear()
        return True

    def get_attack_hitbox(self):
        # generate a rectangle in front of the fighter depending on attack type
        reach = 28 if self.attack_type == "light" else 44
        width = 24 if self.attack_type == "light" else 36
        height = 28
        if self.facing == 1:
            ax = self.rect.right + reach - width
        else:
            ax = self.rect.left - (reach - width + 0)
        ay = self.rect.centery - height // 2
        return pygame.Rect(ax, ay, width, height)

    def take_hit(self, dmg, knockback, attacker_facing):
        if self.invincible_timer > 0:
            return False
        # reduce if blocking and facing attacker
        block_effective = self.is_blocking and (self.facing == -attacker_facing)
        if block_effective:
            dmg = math.ceil(dmg * BLOCK_REDUCTION)
            # smaller knockback if blocked
            knockback = knockback * 0.4
        self.hp = max(0, self.hp - dmg)
        # apply knockback
        self.vx += knockback * attacker_facing * -1
        # i-frames
        self.invincible_timer = INVINCIBLE_FRAMES
        return True

    def physics_only_update(self):
        # decrement timers
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.attack_timer > 0:
            self.attack_timer -= 1
            if self.attack_timer <= 0:
                self.is_attacking = False
                self.attack_type = None
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

        # physics
        self.apply_physics()

    def draw(self, surf):
        # flicker if invincible
        if self.invincible_timer % 4 < 2 and self.invincible_timer > 0:
            # Draw with transparency for invincibility effect
            if self.gif_image:
                image = self.gif_image.get_image(flip=(self.facing == -1))
                image.set_alpha(128)
                # Center the sprite on the fighter position
                sprite_rect = image.get_rect(center=(self.rect.centerx, self.rect.centery - 10))
                surf.blit(image, sprite_rect)
            else:
                outline = (180,180,180)
                pygame.draw.rect(surf, outline, self.rect.inflate(4,4))
        else:
            if self.gif_image:
                image = self.gif_image.get_image(flip=(self.facing == 1))
                # Center the sprite on the fighter position
                sprite_rect = image.get_rect(center=(self.rect.centerx, self.rect.centery - 10))
                surf.blit(image, sprite_rect)
            else:
                pygame.draw.rect(surf, self.color, self.rect)

    def is_alive(self):
        return self.hp > 0

# ---------- Player (human) with Blue Fireballs ----------
class Player(Fighter):
    def __init__(self, x, color, controls, gif_file=None, scale=1.0):
        super().__init__(x, color, gif_file, scale)
        self.controls = controls
        self.fireballs = []
        self.fire_cooldown = 0
        self.blue_fire_cooldown = 0

    def update(self, pressed_keys):
        # decrement timers & physics handled in base helper
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.attack_timer > 0:
            self.attack_timer -= 1
            if self.attack_timer <= 0:
                self.is_attacking = False
                self.attack_type = None
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

        # Update fireballs
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.blue_fire_cooldown > 0:  # ADD THIS LINE
            self.blue_fire_cooldown -= 1  # ADD THIS LINE
            
        for fireball in self.fireballs[:]:
            fireball.update()
            if not fireball.active:
                self.fireballs.remove(fireball)

        # inputs
        left = pressed_keys[self.controls['left']]
        right = pressed_keys[self.controls['right']]
        jump = pressed_keys[self.controls['jump']]
        light = pressed_keys[self.controls['light']]
        heavy = pressed_keys[self.controls['heavy']]
        block = pressed_keys[self.controls['block']]

        # blocking has priority (can't move much while blocking)
        self.is_blocking = block and not self.is_attacking

        if not self.is_blocking:
            # horizontal movement
            if left and not right:
                self.vx = -self.speed
                self.facing = -1
            elif right and not left:
                self.vx = self.speed
                self.facing = 1
            else:
                self.vx *= 0.7  # Faster stop when no input

            # jump
            if jump and self.on_ground:
                self.vy = self.jump_strength
                self.on_ground = False

            # attacks - with increased cooldown to prevent spam
            if light and not self.is_attacking and self.attack_cooldown == 0:
                self.start_attack("light")
                self.attack_cooldown=40
            elif heavy and not self.is_attacking and self.attack_cooldown == 0 and self.blue_fire_cooldown == 0:
                self.start_attack("heavy")
                self.breath_blue_fire()

        # physics
        self.apply_physics()
        
        # Update animation state
        self.physics_only_update()

    def breath_blue_fire(self):
        if self.fire_cooldown <= 0:
            fireball = BlueFireball(
                self.rect.centerx, 
                self.rect.centery,
                self.facing,
                speed=5
            )
            self.fireballs.append(fireball)
            self.fire_cooldown = 30
            self.blue_fire_cooldown = 5 * FPS 

    def draw(self, surf):
        # Draw blue fireballs first
        for fireball in self.fireballs:
            fireball.draw(surf)
            
        # Draw player using parent method
        super().draw(surf)

# ---------- Enhanced Dragon AI with Advanced Learning ----------
class DragonAI(Fighter):
    def __init__(self, x, color, target: Fighter, gif_file=None, scale=1.0, model_file="ai_model.pkl"):
        super().__init__(x, color, gif_file, scale)
        self.target = target
        self.model_file = model_file
        
        # Enhanced Q-learning parameters
        self.learning_rate = 0.15
        self.discount_factor = 0.98
        self.exploration_rate = 0.5  # Higher initial exploration
        self.exploration_decay = 0.998
        self.min_exploration = 0.02
        
        # Expanded action spaces with more strategic options
        self.actions = [
            "idle", "approach_fast", "approach_slow", "retreat_fast", "retreat_slow",
            "attack_light", "attack_heavy", "dodge_left", "dodge_right", "jump_evade",
            "jump_attack", "wait_pattern", "counter_attack"
        ]
        
        # Enhanced state tracking
        self.last_state = None
        self.last_action = None
        self.total_reward = 0
        self.match_count = 0
        self.win_patterns = set()
        
        # Decision making
        self.decision_cooldown = 0
        self.current_intent = "idle"
        self.pattern_memory = deque(maxlen=20)  # Remember recent game states
        
        # Load or initialize Q-table
        self.q_table = self.load_model()
        if self.q_table is None:
            self.q_table = {}
            
    def get_state(self):
        """Enhanced state representation with more game features"""
        # Distance to target (more granular)
        dist = self.target.x - self.x
        abs_dist = abs(dist)
        
        if abs_dist < 40:
            dist_state = "very_close"
        elif abs_dist < 100:
            dist_state = "close"
        elif abs_dist < 200:
            dist_state = "medium"
        else:
            dist_state = "far"
            
        # Relative HP with more states
        hp_ratio = self.hp / self.max_hp
        target_hp_ratio = self.target.hp / self.target.max_hp
        
        if hp_ratio > 0.8:
            hp_state = "very_high"
        elif hp_ratio > 0.6:
            hp_state = "high"
        elif hp_ratio > 0.4:
            hp_state = "medium"
        elif hp_ratio > 0.2:
            hp_state = "low"
        else:
            hp_state = "very_low"
            
        if target_hp_ratio > 0.8:
            target_hp_state = "very_high"
        elif target_hp_ratio > 0.6:
            target_hp_state = "high"
        elif target_hp_ratio > 0.4:
            target_hp_state = "medium"
        elif target_hp_ratio > 0.2:
            target_hp_state = "low"
        else:
            target_hp_state = "very_low"
            
        # Enhanced target state tracking
        target_attacking = "attacking" if self.target.is_attacking else "not_attacking"
        target_blocking = "blocking" if self.target.is_blocking else "not_blocking"
        target_airborne = "airborne" if not self.target.on_ground else "grounded"
        
        # Fireball detection
        fireball_threat = "no_fireball"
        for fireball in self.target.fireballs:
            if (abs(fireball.x - self.x) < 150 and 
                abs(fireball.y - self.y) < 100):
                fireball_threat = "fireball_close"
                break
        
        # Self state
        self_attacking = "attacking" if self.is_attacking else "not_attacking"
        self_blocking = "blocking" if self.is_blocking else "not_blocking"
        on_ground_state = "grounded" if self.on_ground else "airborne"
        
        # Create enhanced state key
        state_key = (
            dist_state, hp_state, target_hp_state,
            target_attacking, target_blocking, target_airborne,
            fireball_threat, self_attacking, self_blocking, on_ground_state
        )
        
        # Store in pattern memory
        self.pattern_memory.append(state_key)
        
        return state_key
        
    def choose_action(self, state):
        """Enhanced action selection with pattern recognition"""
        # Initialize state in Q-table if not present
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
            
        # Check for winning patterns
        if self.detect_winning_pattern():
            best_action = self.get_pattern_based_action()
            if best_action:
                return best_action
            
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: choose random action with some bias towards good ones
            if random.random() < 0.3 and "fireball_close" in state:
                return random.choice(["jump_evade", "dodge_left", "dodge_right"])
            return random.choice(self.actions)
        else:
            # Exploit: choose best action
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
            
    def detect_winning_pattern(self):
        """Detect repeating patterns that lead to wins"""
        if len(self.pattern_memory) < 5:
            return False
            
        # Simple pattern detection - look for repeated state sequences
        recent_states = list(self.pattern_memory)[-5:]
        return len(set(recent_states)) < 3  # If states are repeating
        
    def get_pattern_based_action(self):
        """Get action based on detected patterns"""
        # Analyze recent states to choose counter actions
        recent_states = list(self.pattern_memory)[-3:]
        
        for state in recent_states:
            if "fireball_close" in state:
                return random.choice(["jump_evade", "dodge_left"])
            if "attacking" in state and "close" in state:
                return "counter_attack"
            if "very_low" in state and "close" in state:
                return "attack_heavy"
                
        return None
            
    def update_q_value(self, state, action, reward, next_state):
        """Enhanced Q-learning with momentum"""
        # Initialize next state if not present
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
            
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state
        max_next_q = max(self.q_table[next_state].values())
        
        # Enhanced Q-learning update with adaptive learning
        adaptive_lr = self.learning_rate * (1 + self.match_count * 0.01)
        new_q = current_q + adaptive_lr * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Store winning patterns
        if reward >= 50:  # Significant positive reward
            self.win_patterns.add((state, action))
        
    def get_reward(self, state, action, next_state):
        """Enhanced reward system with more strategic considerations"""
        reward = 0
        
        # Reward for dealing damage
        hp_difference = (self.last_hp - self.hp) if hasattr(self, 'last_hp') else 0
        target_hp_difference = (self.last_target_hp - self.target.hp) if hasattr(self, 'last_target_hp') else 0
        
        # Positive reward for damaging opponent
        reward += target_hp_difference * 15
        
        # Negative reward for taking damage
        reward -= hp_difference * 20
        
        # Strategic rewards
        if action in ["jump_evade", "dodge_left", "dodge_right"] and "fireball_close" in state:
            # Reward for successfully dodging fireballs
            reward += 10
            
        if action == "counter_attack" and self.target.is_attacking:
            reward += 8
            
        if action in ["approach_fast", "attack_heavy"] and "very_low" in next_state[2]:  # target_hp_state
            reward += 12
            
        # Punish bad decisions
        if action in ["approach_fast", "approach_slow"] and "fireball_close" in state:
            reward -= 15
            
        if action == "idle" and "very_close" in state and not self.target.is_attacking:
            reward -= 5
            
        # Large reward for winning
        if not self.target.is_alive():
            reward += 200
            self.match_count += 1
            
        # Large penalty for losing
        if not self.is_alive():
            reward -= 200
            self.match_count += 1
            
        return reward
        
    def save_model(self):
        """Save enhanced Q-table with patterns"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'exploration_rate': self.exploration_rate,
                    'match_count': self.match_count,
                    'win_patterns': list(self.win_patterns)
                }, f)
            print(f"Enhanced AI model saved with {len(self.q_table)} states and {len(self.win_patterns)} winning patterns")
        except Exception as e:
            print(f"Error saving model: {e}")
            
    def load_model(self):
        """Load enhanced Q-table with patterns"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.exploration_rate = data.get('exploration_rate', 0.5)
                    self.match_count = data.get('match_count', 0)
                    self.win_patterns = set(data.get('win_patterns', []))
                    print(f"Enhanced AI model loaded with {len(data['q_table'])} states, {self.match_count} matches played")
                    return data['q_table']
        except Exception as e:
            print(f"Error loading model: {e}")
        return None
        
    def decide(self):
        """Enhanced decision making with pattern analysis"""
        state = self.get_state()
        action = self.choose_action(state)
        
        # Store previous state for learning
        if self.last_state is not None and self.last_action is not None:
            reward = self.get_reward(self.last_state, self.last_action, state)
            self.total_reward += reward
            self.update_q_value(self.last_state, self.last_action, reward, state)
            
        self.last_state = state
        self.last_action = action
        self.last_hp = self.hp
        self.last_target_hp = self.target.hp
        
        self.current_intent = action
        self.decision_cooldown = random.randint(10, 25)
        
        # Adaptive exploration decay
        decay_rate = self.exploration_decay
        if self.match_count > 50:
            decay_rate = 0.999  # Slower decay for experienced AI
            
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * decay_rate)
        
    def update(self):
        # Update timers
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        if self.attack_timer > 0:
            self.attack_timer -= 1
            if self.attack_timer <= 0:
                self.is_attacking = False
                self.attack_type = None
        if self.invincible_timer > 0:
            self.invincible_timer -= 1

        # Make decision if cooldown is up
        if self.decision_cooldown <= 0:
            self.decide()
        else:
            self.decision_cooldown -= 1

        # Execute current intent with enhanced behaviors
        self.is_blocking = (self.current_intent == "counter_attack") and not self.is_attacking

        if not self.is_blocking:
            # Enhanced movement based on intent
            if self.current_intent == "approach_fast":
                self.vx = self.speed * 1.5 * (1 if self.target.x > self.x else -1)
                self.facing = 1 if self.target.x > self.x else -1
            elif self.current_intent == "approach_slow":
                self.vx = self.speed * 0.7 * (1 if self.target.x > self.x else -1)
                self.facing = 1 if self.target.x > self.x else -1
            elif self.current_intent == "retreat_fast":
                self.vx = -self.speed * 1.5 * (1 if self.target.x > self.x else -1)
                self.facing = -1 if self.target.x > self.x else 1
            elif self.current_intent == "retreat_slow":
                self.vx = -self.speed * 0.7 * (1 if self.target.x > self.x else -1)
                self.facing = -1 if self.target.x > self.x else 1
            elif self.current_intent == "dodge_left":
                self.vx = -self.speed * 2
                self.facing = -1
            elif self.current_intent == "dodge_right":
                self.vx = self.speed * 2
                self.facing = 1
            elif self.current_intent in ["jump_evade", "jump_attack"] and self.on_ground:
                self.vy = self.jump_strength
                self.on_ground = False
                if self.current_intent == "jump_attack" and random.random() < 0.7:
                    self.start_attack("heavy")
            elif self.current_intent == "wait_pattern":
                self.vx *= 0.3  # Almost stop
                
            # Attacks
            if self.current_intent == "attack_light":
                self.start_attack("light")
            elif self.current_intent == "attack_heavy":
                self.start_attack("heavy")
            elif self.current_intent == "counter_attack" and not self.is_attacking:
                if random.random() < 0.8:
                    self.start_attack("heavy")

        # Smart facing
        if self.x < self.target.x:
            self.facing = 1
        else:
            self.facing = -1

        # Physics
        self.apply_physics()
        self.physics_only_update()
        
    def reset_after_match(self):
        """Reset AI state after a match"""
        # Save model periodically
        if self.match_count % 3 == 0:
            self.save_model()
            
        self.last_state = None
        self.last_action = None
        self.total_reward = 0
        self.pattern_memory.clear()

# ---------- Simple HUD ----------
def draw_hp_bar(surf, x, y, w, h, current, maxv, color):
    pygame.draw.rect(surf, (80,80,80), (x, y, w, h))
    ratio = current / maxv
    inner_w = max(0, int((w-4) * ratio))
    pygame.draw.rect(surf, color, (x+2, y+2, inner_w, h-4))
    # border
    pygame.draw.rect(surf, (20,20,20), (x,y,w,h), 2)

# ---------- Background and Environment ----------
class Background:
    def __init__(self):
        self.layers = []
        # Create a simple multi-layer background
        for i in range(3):
            layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            # Draw some background elements (mountains, clouds, etc.)
            color = (100 - i*20, 120 - i*20, 150 - i*10)
            if i == 0:  # Far mountains
                for j in range(5):
                    x = j * 200
                    height = random.randint(80, 120)
                    pygame.draw.polygon(layer, color, [
                        (x, HEIGHT), 
                        (x+100, HEIGHT - height),
                        (x+200, HEIGHT)
                    ])
            elif i == 1:  # Mid mountains
                for j in range(8):
                    x = j * 150
                    height = random.randint(100, 150)
                    pygame.draw.polygon(layer, color, [
                        (x, HEIGHT), 
                        (x+75, HEIGHT - height),
                        (x+150, HEIGHT)
                    ])
            else:  # Near hills
                for j in range(12):
                    x = j * 100
                    height = random.randint(30, 60)
                    pygame.draw.polygon(layer, color, [
                        (x, HEIGHT), 
                        (x+50, HEIGHT - height),
                        (x+100, HEIGHT)
                    ])
            self.layers.append(layer)
            
    def draw(self, surf):
        # Draw sky
        surf.fill((120, 150, 180))
        
        # Draw background layers
        for i, layer in enumerate(self.layers):
            # Parallax effect based on layer
            offset = i * 10
            surf.blit(layer, (-offset, 0))
            
        # Draw ground
        pygame.draw.rect(surf, (40,100,40), (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        # Ground details
        for i in range(20):
            x = i * 50
            pygame.draw.rect(surf, (30,90,30), (x, GROUND_Y, 25, 10))

# ---------- Sound System ----------
class SoundManager:
    def __init__(self):
        self.sounds = {}
        # Create placeholder sounds (in a real game, you'd load actual sound files)
        try:
            # These would be actual sound files in a real implementation
            # self.sounds["hit"] = pygame.mixer.Sound("hit.wav")
            # self.sounds["attack"] = pygame.mixer.Sound("attack.wav")
            # self.sounds["fire"] = pygame.mixer.Sound("fire.wav")
            pass
        except:
            print("Sound files not found, continuing without sound")
            
    def play(self, sound_name):
        if sound_name in self.sounds:
            self.sounds[sound_name].play()

# ---------- Main Game ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Dragon vs Knight — Enhanced Learning AI Combat")
    
    # Initialize sound system
    sound_manager = SoundManager()
    
    # Create background
    background = Background()
    
    # Create fighters with arrow key controls for knight
    p1_controls = {
        'left': K_LEFT, 'right': K_RIGHT, 'jump': K_UP,
        'light': K_DOWN, 'heavy': K_RCTRL, 'block': K_RSHIFT
    }
    
    p1 = Player(120, (70,140,230), p1_controls,
                gif_file="assets/knight.gif",
                scale=0.5)
    
    p2 = DragonAI(WIDTH - 160, (220,90,80), target=p1,
                  gif_file="assets/dragon2.png",
                  scale=0.18,
                  model_file="enhanced_ai_model.pkl")
    
    # make p2 face left initially
    p2.facing = -1

    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 38)
    round_time = 60 * FPS  # 60 seconds
    timer = round_time

    # simple match reset function
    def reset_match():
        p1.x, p1.y = 120, GROUND_Y - PLAYER_HEIGHT
        p2.x, p2.y = WIDTH - 160, GROUND_Y - PLAYER_HEIGHT
        p1.hp = p1.max_hp
        p2.hp = p2.max_hp
        p1.vx = p1.vy = 0
        p2.vx = p2.vy = 0
        p1.invincible_timer = p2.invincible_timer = 0
        p1.fireballs.clear()
        p2.current_intent = "idle"
        p2.decision_cooldown = 0
        p2.reset_after_match()

    running = True
    paused = False
    game_over = False

    while running:
        dt = clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == QUIT:
                # Save AI model before quitting
                p2.save_model()
                running = False
            if ev.type == KEYDOWN and ev.key == K_ESCAPE:
                # Save AI model before quitting
                p2.save_model()
                running = False
            if ev.type == KEYDOWN and ev.key == K_p:
                paused = not paused
            if ev.type == KEYDOWN and ev.key == K_r:
                reset_match()
                timer = round_time
                game_over = False

        if paused:
            screen.fill((40,40,50))
            pause_text = big_font.render("PAUSED - press P to resume", True, (240,240,240))
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - 20))
            pygame.display.flip()
            continue
            
        if game_over:
            keys = pygame.key.get_pressed()
            if keys[K_r]:
                reset_match()
                timer = round_time
                game_over = False
            else:
                pygame.time.wait(100)
            continue

        pressed = pygame.key.get_pressed()

        # update facing
        if p1.x < p2.x:
            p1.facing = 1
            p2.facing = -1
        else:
            p1.facing = -1
            p2.facing = 1

        # update fighters
        p1.update(pressed)
        p2.update()

        # handle attack collisions
        for attacker, defender in ((p1, p2), (p2, p1)):
            if attacker.is_attacking:
                hb = attacker.get_attack_hitbox()
                if hb.colliderect(defender.rect):
                    if id(defender) not in attacker.attack_hit_targets:
                        dmg = LIGHT_DAMAGE if attacker.attack_type == "light" else HEAVY_DAMAGE
                        attacked = defender.take_hit(dmg, knockback=6 if attacker.attack_type=="heavy" else 3, attacker_facing=attacker.facing)
                        if attacked:
                            attacker.attack_hit_targets.add(id(defender))
                            sound_manager.play("hit")

        # handle blue fireball collisions
        for fireball in p1.fireballs[:]:
            if fireball.get_rect().colliderect(p2.rect):
                if p2.take_hit(FIRE_DAMAGE, knockback=4, attacker_facing=p1.facing):
                    p1.fireballs.remove(fireball)
                    sound_manager.play("fire")

        # check round end conditions
        if not p1.is_alive() or not p2.is_alive() or timer <= 0:
            game_over = True
            if p1.hp == p2.hp:
                winner_text = "Draw!"
            elif p1.hp > p2.hp:
                winner_text = "Knight Wins!"
            else:
                winner_text = "Dragon Wins!"
            screen.fill((30,30,40))
            win_render = big_font.render(winner_text, True, (240,240,240))
            screen.blit(win_render, (WIDTH//2 - win_render.get_width()//2, HEIGHT//2 - 40))
            
            # Show enhanced AI learning stats
            stats_text = font.render(f"Dragon Matches: {p2.match_count}, Learning: {p2.exploration_rate:.3f}", True, (200,200,200))
            screen.blit(stats_text, (WIDTH//2 - stats_text.get_width()//2, HEIGHT//2))
            
            patterns_text = font.render(f"Winning Patterns Learned: {len(p2.win_patterns)}", True, (200,200,200))
            screen.blit(patterns_text, (WIDTH//2 - patterns_text.get_width()//2, HEIGHT//2 + 20))
            
            info = font.render("Press R to restart, ESC to quit", True, (200,200,200))
            screen.blit(info, (WIDTH//2 - info.get_width()//2, HEIGHT//2 + 50))
            pygame.display.flip()
            
            # Save AI model after each match
            p2.save_model()
            continue

        # decrement timer
        timer -= 1

        # ---------- Draw ----------
        background.draw(screen)

        # draw fighters with fireballs
        p1.draw(screen)
        p2.draw(screen)

        # HUD
        draw_hp_bar(screen, 20, 20, 300, 28, p1.hp, p1.max_hp, (70,140,230))
        draw_hp_bar(screen, WIDTH - 320, 20, 300, 28, p2.hp, p2.max_hp, (220,90,80))
        time_s = timer // FPS
        time_text = font.render(f"Time: {time_s}s", True, (20,20,20))
        screen.blit(time_text, (WIDTH//2 - 30, 24))

        # enhanced labels with controls and AI info
        p1_label = font.render("Knight: ARROWS DOWN=Light RCTRL=Fire RSHIFT=Block", True, (240,240,240))
        ai_label = font.render(f"Dragon: {p2.current_intent} (Matches: {p2.match_count}, Patterns: {len(p2.win_patterns)})", True, (240,240,240))
        screen.blit(p1_label, (10, HEIGHT - 40))
        screen.blit(ai_label, (10, HEIGHT - 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
