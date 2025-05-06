import csv
import time
import os

# **Oyun verilerinin kaydedileceği CSV dosyası**
csv_filename = "game_data.csv"

# **CSV dosyasını başlat (başlıkları ekle)**
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pacman_x", "pacman_y", "ghost_x", "ghost_y", "score", "reward"])

def save_game_data(pacman_x, pacman_y, ghost_x, ghost_y, score, reward):
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([pacman_x, pacman_y, ghost_x, ghost_y, score, reward])
    #print(f"Kayıt: P({pacman_x}, {pacman_y}) - G({ghost_x}, {ghost_y}) - Score: {score}")

# **Pygame ile bağlantıyı kontrol etmek için test fonksiyonu**
if __name__ == "__main__":
    print("Pac-Man verileri kaydedilmeye başlandı!")
    while True:
        # Simülasyon için rastgele veri (gerçek oyundan çekmelisin!)
        pacman_x, pacman_y = 100, 150  # Gerçekte oyundan gelen değer olmalı
        ghost_x, ghost_y = 200, 250    # Gerçekte oyundan gelen değer olmalı
        score = 5                      # Gerçekte oyundan gelen değer olmalı
        
        save_game_data(pacman_x, pacman_y, ghost_x, ghost_y, score)
        
        time.sleep(0.1)  # 0.1 saniyede bir kayıt (gerektiğinde hızını değiştirebilirsin)
