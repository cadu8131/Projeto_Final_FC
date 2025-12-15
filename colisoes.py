import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mudar de acordo com a máquina do usuário
torch.manual_seed(42)
np.random.seed(42)
# Para reprodução dos resultados

def gerar_dados_rebote(n_samples=2000, nx=256, nt=256):
    """
    Simula uma colisão (Scattering).
    """
    
    x = np.linspace(-15, 15, nx)
    t = np.linspace(0, 10, nt)
    X, T = np.meshgrid(x, t)
    
    mapas = []
    velocidades = []
    
    # Centro da colisão no tempo
    tc = 5.0 

    for _ in range(n_samples):
        v = np.random.uniform(0.2, 0.8)
        # Ansartz da distância mínima, modelo hiperbolico
        distancia_minima = 2.0 * (1.0 - v) # Quanto mais rápido, mais perto chegam
        
        # Trajetória do Kink (Lado Esquerdo)
        traj_pos = np.sqrt(distancia_minima**2 + v**2 * (T - tc)**2)
        
        # Trajetória do Antikink (Lado Direito - Simétrico)
        traj_neg = -traj_pos
        
        # Ansartz inicial
        phi = np.tanh(X - traj_neg) - np.tanh(X - traj_pos) - 1.0
        
        mapas.append(phi)
        velocidades.append(v)

    # Converter para Tensor [N, 1, Nt, Nx] - Precisaremos dele para solução
    mapas_tensor = torch.tensor(np.array(mapas), dtype=torch.float32).unsqueeze(1)
    vel_tensor = torch.tensor(np.array(velocidades), dtype=torch.float32).unsqueeze(1)
    
    return vel_tensor, mapas_tensor

entradas_v, alvos_mapa = gerar_dados_rebote(n_samples=2000)

class PhysicsGenerator(nn.Module):
    # Modelo treinável
    # O que o modelo faz é criar um mapa que faz a operação
    # velocidade inicial -> campo(x,t).
    def __init__(self):
        super(PhysicsGenerator, self).__init__()
        
        # Entra um tensor, pois iremos realizar convoluções, que trabalham com volumes
        self.fc = nn.Linear(1, 128 * 4 * 4) 
        
        
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), 
            # Faz a convolução para crescer o tamanho do nosso tensor, cada uso faz
            ## dobrar a resolução, para termos um melhor mapa de cores no futuro
            nn.BatchNorm2d(64),
            # Estabilizador de treino
            nn.LeakyReLU(0.2),
            # Evita regiões onde a solução trivial aparece -> campo sempre zero
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(4, 1, 4, 2, 1),
            nn.Tanh() 
            # Recuperando o campo físico
        )

    def forward(self, v):
        x = self.fc(v)
        x = x.view(-1, 128, 4, 4)
        return self.decoder(x)

model = PhysicsGenerator().to(device)
# Criando modelo

criterio = nn.L1Loss() 
# Erro absoluto: estamos importando a função | \phi_real - \phi_pred |

optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
# O Adam resolve o problema.

dataset = torch.utils.data.TensorDataset(entradas_v, alvos_mapa)
# Pegando os dados já gerados
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# Transformando o dataset

print("Iniciando treinamento")
for epoch in range(30):
# Vamos trabalhar com 30 épocas
    model.train()
    running_loss = 0.0
    for v, map_target in dataloader:
        v, map_target = v.to(device), map_target.to(device)
        
        optimizer.zero_grad()
        output = model(v)
        loss = criterio(output, map_target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        print(f"Época {epoch+1}: Loss = {running_loss/len(dataloader):.6f}")


model.eval()
# Testar velocidade
test_vs = torch.tensor([[0.8]], dtype=torch.float32).to(device)

with torch.no_grad():
    gen_maps = model(test_vs).cpu().numpy()

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(
    gen_maps.squeeze(),
    cmap='RdBu',
    origin='lower',
    aspect='auto',
    vmin=-1.2,
    vmax=1.2
)

ax.set_xlabel("index x")
ax.set_ylabel("index t")

# Linha do momento da colisão (t ≈ 128)
ax.axhline(128, linestyle='--', alpha=0.5, label='Collision')

ax.legend()
plt.tight_layout()
plt.show()
