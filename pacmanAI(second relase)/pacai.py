
import numpy as np
import random
import pygame

np.seterr(all='warn')  # Taşmaları ve NaN’leri uyarı olarak bildir

# Başlangıç parametreleri
WIDTH, HEIGHT = 600, 600
pacman_speed = 5
ghost_speed = 3

def init_params():
    W1 = np.random.randn(10, 6) * 0.01  # Küçük değerler
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(4, 10) * 0.01
    b2 = np.zeros((4, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    # Z'nin son boyutta olduğunu varsayıyoruz (örneğin (4, 1))
    Z = Z - np.max(Z, axis=0, keepdims=True)  # Her sütun için max çıkar
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X, return_intermediates=False):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    if return_intermediates:
        return A2, Z1, A1
    return A2

def get_action(probs, epsilon=0.1,shey=0.2):
     # Olasılıkları sırala ve indekslerini al
    sorted_indices = np.argsort(probs.flatten())[::-1]  # Büyükten küçüğe sırala
    
    randomP = np.random.rand()
    
    if randomP < shey:
        return sorted_indices[1]  # İkinci en iyi eylem
    elif randomP < (shey + epsilon):
        return sorted_indices[2]  # üçünü en iyi eylem
    else:
        return sorted_indices[0]  # En iyi eylem

def makeplay(action, speed):
    moves = {
        0: (0, -speed),
        1: (0, speed),
        2: (-speed, 0),
        3: (speed, 0)
    }
    return moves.get(action, (0, 0))

def get_state(pacman_x, pacman_y, ghost_x, ghost_y, score):
    max_distance = np.sqrt(WIDTH**2 + HEIGHT**2)
    return np.array([
        [pacman_x / WIDTH],
        [pacman_y / HEIGHT],
        [ghost_x / WIDTH],
        [ghost_y / HEIGHT],
        [score / 10],  # Skoru normalize et
        [np.sqrt((pacman_x - ghost_x)**2 + (pacman_y - ghost_y)**2) / max_distance]
    ])

def discount_rewards(rewards, gamma=0.95):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0
    for t in reversed(range(len(rewards))):
        cumulative = rewards[t] + gamma * cumulative
        discounted[t] = cumulative
    return discounted

def update_policy(states, actions, rewards, W1, b1, W2, b2, learning_rate=0.01):
    
    discounted = discount_rewards(rewards)
    mean = np.mean(discounted)
    std = np.std(discounted)
    if std != 0:
        discounted = (discounted - mean) / std

    for i in range(len(states)):
        X = states[i]
        A2, Z1, A1 = forward_prop(W1, b1, W2, b2, X, return_intermediates=True)
        one_hot = np.zeros_like(A2)
        one_hot[actions[i]] = 1

        dZ2 = A2 - one_hot
        dW2 = dZ2 @ A1.T * discounted[i]
        db2 = dZ2 * discounted[i]

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (Z1 > 0)
        dW1 = dZ1 @ X.T * discounted[i]
        db1 = dZ1 * discounted[i]

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return W1, b1, W2, b2

def train_ai(W1, b1, W2, b2, episodes=1000):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pacman RL")
    clock = pygame.time.Clock()

    BLACK = (0, 0, 0)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    points = [(random.randint(20, 580), random.randint(20, 580)) for _ in range(20)]
    for episode in range(episodes):
        
        pacman_x, pacman_y = 300, 300
        ghost_x, ghost_y = 150, 150
        score = 0
        
        

        state = get_state(pacman_x, pacman_y, ghost_x, ghost_y, score)
        states, actions, rewards = [], [], []
        
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            screen.fill(BLACK)

            states.append(state)
            probs = forward_prop(W1, b1, W2, b2, state)
            action = action = get_action(probs,epsilon=0.1,shey=0.2)
            actions.append(action)

            dx, dy = makeplay(action, pacman_speed)
            pacman_x += dx
            pacman_y += dy

            # Ekran dışına çıkmayı önle
            pacman_x = max(10, min(WIDTH - 10, pacman_x))
            pacman_y = max(10, min(HEIGHT - 10, pacman_y))
            ghost_x = max(10, min(WIDTH - 10, ghost_x))
            ghost_y = max(10, min(HEIGHT - 10, ghost_y))

            
            

            # Hayalet hareketi
            ghost_x += ghost_speed if pacman_x > ghost_x else -ghost_speed
            ghost_y += ghost_speed if pacman_y > ghost_y else -ghost_speed

            
            

            reward = 0
            removed_points = []
            for point in points[:]:
                if abs(pacman_x - point[0]) < 10 and abs(pacman_y - point[1]) < 10:
                    points.remove(point)
                    score += 1
                    reward += 10
                    removed_points.append(point)

            
            if abs(pacman_x - ghost_x) < 10 and abs(pacman_y - ghost_y) < 10:
                reward -= 100
                running = False
                points.extend(removed_points)
                 
          

            if not points:
                reward += 100
                print(f"Bölüm {episode+1}: Tüm puanlar toplandı!")
                break

            rewards.append(reward)
            state = get_state(pacman_x, pacman_y, ghost_x, ghost_y, score)

            pygame.draw.circle(screen, YELLOW, (pacman_x, pacman_y), 10)
            pygame.draw.circle(screen, RED, (ghost_x, ghost_y), 10)
            for point in points:
                pygame.draw.circle(screen, GREEN, point, 5)
            

            pygame.display.flip()
            clock.tick(60)

        W1, b1, W2, b2 = update_policy(states, actions, rewards, W1, b1, W2, b2)

        # Loss hesapla
        total_loss = 0
        disc_rewards = discount_rewards(rewards)
        for i in range(len(states)):
            A2 = forward_prop(W1, b1, W2, b2, states[i])
            total_loss += -np.log(A2[actions[i]].item()) * disc_rewards[i]

        print(f"Epizot {episode+1} - Loss: {total_loss:.2f}")
        #print("discounted rewards:", disc_rewards)

        # Eğitim izleme
        if episode % 100 == 0:
            pygame.time.wait(500)

    pygame.quit()
    return W1, b1, W2, b2

# Eğitim başlat
if __name__ == "__main__":
    W1, b1, W2, b2 = init_params()
    train_ai(W1, b1, W2, b2, episodes=1000)
