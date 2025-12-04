import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GRU

def create_sequences(data, timestep, target_col_idx):
    X, Y = [], []
    for i in range(timestep, len(data)):
        X.append(data[i-timestep:i, :])
        Y.append(data[i, target_col_idx])
    return np.array(X), np.array(Y)

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_predictions(df_raw, freq, timestep, model_name, feature_cols, epochs, cutoff='2025-01-01'):
    df_res = df_raw.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    df_train = df_res.loc[:cutoff].iloc[:-1].copy()
    df_test = df_res.loc[cutoff:].copy()

    df_train['volatilidad'] = df_train['high'] - df_train['low']
    df_test['volatilidad'] = df_test['high'] - df_test['low']

    data_train = df_train[feature_cols].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(data_train)

    dataset_full = pd.concat((df_train[feature_cols].iloc[-timestep:], df_test[feature_cols]), axis=0)
    scaled_test = scaler.transform(dataset_full.values)

    close_idx = feature_cols.index('close')
    X_train, Y_train = create_sequences(scaled_train, timestep, close_idx)
    X_test, _ = create_sequences(scaled_test, timestep, close_idx)

    if os.path.exists(model_name):
        print(f"Cargando modelo existente: {model_name}")
        model = load_model(model_name)
    else:
        print(f"Entrenando nuevo modelo: {model_name} con {epochs} epochs")
        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
        model.save(model_name)

    preds_scaled = model.predict(X_test)
    
    dummy = np.zeros((len(preds_scaled), len(feature_cols)))
    dummy[:, close_idx] = preds_scaled[:, 0]
    preds_real = scaler.inverse_transform(dummy)[:, close_idx]

    return df_test.index[:len(preds_real)], df_test['close'].values[:len(preds_real)], preds_real

try:
    df = pd.read_csv("btc_data.csv", parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = df.loc['2020-01-01':]
except FileNotFoundError:
    print("Error: 'btc_data.csv' no encontrado.")
    exit()

features = ['open', 'high', 'low', 'close', 'volume', 'volatilidad']
configs = [
    {'freq': 'D', 'ts': 60, 'file': 'model_daily.keras', 'label': 'Diario', 'epochs': 25},
    {'freq': 'h', 'ts': 60, 'file': 'model_hourly.keras', 'label': 'Horario', 'epochs': 25},
    {'freq': 'min', 'ts': 60, 'file': 'model_minute.keras', 'label': 'Minuto', 'epochs': 1}
]

results = {}

for cfg in configs:
    try:
        idx, real, pred = get_predictions(df, cfg['freq'], cfg['ts'], cfg['file'], features, cfg['epochs'])
        results[cfg['label']] = {'idx': idx, 'real': real, 'pred': pred}
    except Exception as e:
        print(f"No se pudo procesar frecuencia {cfg['label']}: {e}")

plt.figure(figsize=(15, 8))

if 'Diario' in results:
    base_idx = results['Diario']['idx']
    mask_enero = (base_idx >= '2025-01-01') & (base_idx <= '2025-01-31')
    plt.plot(base_idx[mask_enero], results['Diario']['real'][mask_enero], label='Precio Real (Close)', color='black', linewidth=2)

colors = {'Diario': 'blue', 'Horario': 'green', 'Minuto': 'red'}

for label, data in results.items():
    idx = data['idx']
    pred = data['pred']
    mask = (idx >= '2025-01-01') & (idx <= '2025-01-31')
    
    if mask.sum() > 0:
        plt.plot(idx[mask], pred[mask], label=f'Pred GRU {label}', color=colors.get(label, 'orange'), linestyle='--', linewidth=1, alpha=0.8)

df_ens = pd.DataFrame()
if all(k in results for k in ['Diario', 'Horario', 'Minuto']):
    s_d = pd.Series(results['Diario']['pred'], index=results['Diario']['idx'])
    s_h = pd.Series(results['Horario']['pred'], index=results['Horario']['idx'])
    s_m = pd.Series(results['Minuto']['pred'], index=results['Minuto']['idx'])
    
    start_dt = max(s_d.index.min(), s_h.index.min(), s_m.index.min())
    end_dt = min(s_d.index.max(), s_h.index.max(), s_m.index.max())
    
    s_d = s_d[start_dt:end_dt].resample('min').ffill()
    s_h = s_h[start_dt:end_dt].resample('min').ffill()
    s_m = s_m[start_dt:end_dt].resample('min').ffill()
    
    df_ens = pd.DataFrame({'D': s_d, 'H': s_h, 'M': s_m}).dropna()
    df_ens['Weighted'] = (df_ens['D'] * 0.50) + (df_ens['H'] * 0.30) + (df_ens['M'] * 0.20) * 1.15
    
    mask_ens = (df_ens.index >= '2025-01-01') & (df_ens.index <= '2025-01-31')
    plt.plot(df_ens.index[mask_ens], df_ens['Weighted'][mask_ens], label='Promedio Ponderado', color='magenta', linewidth=2.5)

plt.title('Comparativa Modelos GRU Bitcoin - Enero 2025')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

class Trader:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy 
        self.balance = 100000.0
        self.btc = 0.0
        self.history = []
        self.max_debt = 500000.0
        self.is_locked = False

    @property
    def current_debt(self):
        return abs(min(0, self.balance))

    def execute(self, action, current_price, angle):
        fee = current_price * 0.001 
        
        if self.strategy == 'opt' and self.is_locked and angle < 0:
            if self.btc > 0:
                revenue = (self.btc * current_price) - fee
                self.balance += revenue
                self.btc = 0
            
            current_debt_val = self.current_debt
            if current_debt_val > 0:
                debt_payment = current_debt_val / 2
                self.balance += debt_payment
                
            self.is_locked = False
            
            net_worth = self.balance + (self.btc * current_price)
            self.history.append(net_worth)
            return

        excess_angle = abs(angle) - 15
        if excess_angle < 0: excess_angle = 0
        btc_qty = 1.0 + (np.log1p(excess_angle) * 1.5)

        if action == 'buy':
            if self.is_locked: return 

            cost_total = (btc_qty * current_price) + fee
            available_funds = self.balance + self.max_debt
            
            if available_funds <= 0: return

            if cost_total > available_funds:
                btc_qty = available_funds / current_price
                cost_total = available_funds
            
            self.balance -= cost_total
            self.btc += btc_qty

            if self.current_debt >= (self.max_debt * 0.99):
                self.is_locked = True

        elif action == 'sell':
            if self.btc > 0:
                revenue = (self.btc * current_price) - fee
                self.balance += revenue
                self.btc = 0
                
                if self.is_locked and self.current_debt < (self.max_debt / 2):
                    self.is_locked = False

        net_worth = self.balance + (self.btc * current_price)
        self.history.append(net_worth)

if df_ens.empty:
    print("No se puede simular: Falta el DataFrame del promedio ponderado.")
    exit()

s_real_min = pd.Series(results['Minuto']['real'], index=results['Minuto']['idx'])
df_ens['Real_Price'] = s_real_min.reindex(df_ens.index).ffill()

traders = [
    Trader("Apostador Optimista", "opt"), 
    Trader("Apostador Pesimista", "pes")
]

look_ahead = 5       
step = 5             
y_scale_factor = 50.0 

print(f"\n--- Iniciando Simulación (Step: {step} min) ---")

df_sim = df_ens[df_ens.index >= '2025-01-01'].copy()

for i in range(0, len(df_sim) - look_ahead, step):
    idx_now = i
    idx_future = i + look_ahead

    pred_now = df_sim['Weighted'].iloc[idx_now]
    pred_future = df_sim['Weighted'].iloc[idx_future]
    price_real_execution = df_sim['Real_Price'].iloc[idx_now]
    
    if pd.isna(price_real_execution): continue 

    dy = (pred_future - pred_now) / y_scale_factor
    dx = look_ahead 
    angle = np.degrees(np.arctan(dy / dx))

    if abs(angle) > 15:
        for t in traders:
            action = 'none'
            if t.strategy == 'opt':
                action = 'buy'
            elif t.strategy == 'pes':
                if angle > 0: action = 'buy'
                else: action = 'sell'
            
            t.execute(action, price_real_execution, angle)
    else:
        for t in traders:
            if t.strategy == 'opt' and t.is_locked and angle < 0:
                 t.execute('none', price_real_execution, angle)
            else:
                current_val = t.balance + (t.btc * price_real_execution)
                t.history.append(current_val)

plt.figure(figsize=(15, 7))
raw_dates = df_sim.index[::step]

for t in traders:
    min_len = min(len(raw_dates), len(t.history))
    safe_dates = raw_dates[:min_len]
    safe_history = t.history[:min_len]

    final_worth = safe_history[-1]
    final_debt = t.current_debt
    
    label_text = (f'{t.name}\n'
                  f'  -> Patr: ${final_worth:,.0f} | Deuda: ${final_debt:,.0f}'
                  f'{" [BLOQ]" if t.is_locked else ""}')
    
    plt.plot(safe_dates, safe_history, label=label_text, linewidth=2)

plt.title(f'Simulación Enero 2025: Logarítmica con Deuda (Protegido)')
plt.ylabel('Patrimonio Neto (USD)')
plt.xlabel('Fecha')
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



if df_ens.empty:
    print("No se puede simular: Falta el DataFrame del promedio ponderado.")
    exit()

# Preparar precio real alineado
s_real_min = pd.Series(results['Minuto']['real'], index=results['Minuto']['idx'])
df_ens['Real_Price'] = s_real_min.reindex(df_ens.index).ffill()

# Instanciar Traders
traders = [
    Trader("Apostador Optimista", "opt"), 
    Trader("Apostador Pesimista", "pes")
]

# Configuración Simulación
look_ahead = 5       
step = 5             
y_scale_factor = 50.0 

print(f"\n--- Iniciando Simulación (Step: {step} min) ---")

# CORRECCIÓN AQUÍ: Filtrar estrictamente solo Enero
mask_sim = (df_ens.index >= '2025-01-01') & (df_ens.index <= '2025-01-31')
df_sim = df_ens[mask_sim].copy()

for i in range(0, len(df_sim) - look_ahead, step):
    idx_now = i
    idx_future = i + look_ahead

    pred_now = df_sim['Weighted'].iloc[idx_now]
    pred_future = df_sim['Weighted'].iloc[idx_future]
    price_real_execution = df_sim['Real_Price'].iloc[idx_now]
    
    if pd.isna(price_real_execution): continue 

    dy = (pred_future - pred_now) / y_scale_factor
    dx = look_ahead 
    angle = np.degrees(np.arctan(dy / dx))

    if abs(angle) > 15:
        for t in traders:
            action = 'none'
            if t.strategy == 'opt':
                action = 'buy'
            elif t.strategy == 'pes':
                if angle > 0: action = 'buy'
                else: action = 'sell'
            
            t.execute(action, price_real_execution, angle)
    else:
        # Si no hay acción agresiva, verificar reglas pasivas (como desbloqueo forzado)
        for t in traders:
            if t.strategy == 'opt' and t.is_locked and angle < 0:
                 t.execute('none', price_real_execution, angle)
            else:
                current_val = t.balance + (t.btc * price_real_execution)
                t.history.append(current_val)

# --- 6. GRÁFICO 2: RESULTADOS SIMULACIÓN ---

plt.figure(figsize=(15, 7))
raw_dates = df_sim.index[::step]

for t in traders:
    min_len = min(len(raw_dates), len(t.history))
    safe_dates = raw_dates[:min_len]
    safe_history = t.history[:min_len]

    final_worth = safe_history[-1]
    final_debt = t.current_debt
    
    label_text = (f'{t.name}\n'
                  f'  -> Patr: ${final_worth:,.0f} | Deuda: ${final_debt:,.0f}'
                  f'{" [BLOQ]" if t.is_locked else ""}')
    
    plt.plot(safe_dates, safe_history, label=label_text, linewidth=2)

plt.title(f'Simulación Enero 2025: Logarítmica con Deuda (Protegido)')
plt.ylabel('Patrimonio Neto (USD)')
plt.xlabel('Fecha')
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()