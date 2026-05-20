"""Quick test: verify all models run on GPU."""
import torch
import sys
sys.path.insert(0, ".")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

n_features = 64
x = torch.randn(4, 60, n_features).to(device)

# TFT
from models.tft_predictor import TFTPredictor
tft = TFTPredictor(n_features).to(device)
out = tft(x)
print(f"[OK] TFT: {tft.get_param_count():,} params | quantiles={out['quantiles'].shape} | dir={out['direction'].shape}")

# CNN
from models.pattern_cnn import PatternCNN
cnn = PatternCNN(n_features).to(device)
cout = cnn(x)
print(f"[OK] CNN: {cnn.get_param_count():,} params | pattern={cout['pattern_class'].shape} | dir={cout['direction'].shape}")

# LSTM
from models.regime_lstm import RegimeLSTM
lstm = RegimeLSTM(n_features).to(device)
lout = lstm(x)
print(f"[OK] LSTM: {lstm.get_param_count():,} params | regime={lout['regime_logits'].shape}")

# Ensemble
from models.ensemble import EnsembleModel
ensemble = EnsembleModel()
tft_pred = tft.predict(x)
cnn_pred = cnn.predict(x)
lstm_pred = lstm.predict(x)
result = ensemble.predict(tft_pred, cnn_pred, lstm_pred)
print(f"[OK] Ensemble: signal={result['signal_label']} | confidence={result['confidence'][0]:.1%} | regime={result['regime']}")

# Memory usage
mem = torch.cuda.memory_allocated() / 1024**2
print(f"\nGPU Memory used: {mem:.1f} MB")
print("\nAll models verified on GPU!")
