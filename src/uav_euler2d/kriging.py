import json                      # biblioteca pra ler arquivos JSON (o split_manifest)
import numpy as np               # biblioteca de contas com vetores e matrizes
import pandas as pd              # biblioteca de tabelas
from sklearn.gaussian_process import GaussianProcessRegressor             # o Kriging
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # as peças do kernel

# caminho da pasta do dataset, guardado numa variável pra usar várias vezes
pasta = r'C:\Users\jpvas\AEROFOLIO\dataset_euler_2d\dataset_euler_2d_banco_v2\dataset_euler_2d'

# lê o cases.csv numa tabela: 60 linhas (casos) e 40 colunas
casos = pd.read_csv(pasta + r'\cases.csv')
# fica só com as linhas cujo status é success ou non_converged (tira os 6 failed)
casos = casos[casos['status'].isin(['success', 'non_converged'])]
print('casos com resultado:', len(casos))     # 54

# lê a divisão treino/validação/teste (um dicionário com listas de geometry_id)
split = json.load(open(pasta + r'\split_manifest.json'))

# separa os casos em três tabelas, pela geometria
treino = casos[casos['geometry_id'].isin(split['train'])]   # 38 perfis
valid  = casos[casos['geometry_id'].isin(split['val'])]     # 8 perfis
teste  = casos[casos['geometry_id'].isin(split['test'])]    # 8 perfis, guardados até o fim
print('treino:', len(treino), '| validação:', len(valid), '| teste:', len(teste))

# faixas de cada variável, usadas pra normalizar entre 0 e 1
limites_inf = np.array([0.00, 0.20, 0.08])   # mínimos de m, p, t
limites_sup = np.array([0.04, 0.60, 0.16])   # máximos de m, p, t

def montar(grupo):                           # função: recebe uma tabela, devolve X e Y
    # X: as colunas m, p, t viram matriz e são normalizadas (cada uma entre 0 e 1)
    X = (grupo[['m', 'p', 't']].values - limites_inf) / (limites_sup - limites_inf)
    # Y: as colunas CL e CD viram matriz
    Y = grupo[['CL', 'CD']].values
    return X, Y                              # devolve as duas

X_tr, Y_tr = montar(treino)                  # X_tr: 38 x 3 | Y_tr: 38 x 2
X_va, Y_va = montar(valid)                   # X_va: 8 x 3  | Y_va: 8 x 2

# o kernel: amplitude x curva de sino (um comprimento por variável) + ruído
kernel = ConstantKernel(1.0) * RBF(length_scale=[0.5, 0.5, 0.5]) + WhiteKernel(1e-3)

# cria o Kriging (ainda sem treinar)
modelo_cl = GaussianProcessRegressor(kernel=kernel,
                                     normalize_y=True,       # normaliza também o CL
                                     n_restarts_optimizer=10,  # tenta 10 chutes iniciais e fica com o melhor
                                     random_state=42)          # fixa os sorteios: mesmo resultado sempre
# treina: ajusta amplitude, comprimentos e ruído pra explicar os 38 CLs de treino
modelo_cl.fit(X_tr, Y_tr[:, 0])              # [:, 0] = todas as linhas, coluna 0 (o CL)
print('kernel ajustado do CL:', modelo_cl.kernel_)   # o kernel depois do ajuste

# prevê os 8 perfis da validação, com a incerteza de cada um
previsto, incerteza = modelo_cl.predict(X_va, return_std=True)
real = Y_va[:, 0]                            # os CLs simulados da validação

# imprime perfil por perfil: previsto ± incerteza, simulado e erro
for p, s, r in zip(previsto, incerteza, real):
    print(f'previsto {p:.4f} ± {s:.4f}   simulado {r:.4f}   erro {p - r:+.4f}')

# RMSE: eleva os erros ao quadrado, tira a média e a raiz: o "erro típico"
rmse = np.sqrt(np.mean((previsto - real)**2))
print(f'RMSE do CL na validação: {rmse:.4f}  ({100 * rmse / real.mean():.1f}% do CL médio)')