# Trading-Denoising-Autoencoder (Trading-DAE)

A **denoising autoencoder** tailored for quantitative-finance data.  It learns to strip away market “noise” from raw price/volume series and surfaces a compact latent representation that can be piped into forecasting or anomaly-detection pipelines.

---

## ✨ Key ideas

* **Signal over noise** – Remove microstructure noise, bid–ask bounce, and intraday seasonality.  
* **Latent manifold** – Low-dimensional embeddings capture regime shifts and co-movement.  
* **Modular design** – Swap encoders (1-D CNN, TCN, Transformer) and decoders with a single YAML edit. Currently only 1D CNN is implemented
* **Back-test friendly** – End-to-end reproducible experiments that integrate with Backtesting.py
