
## Introduction

- The notebooks for training demonstrations and the definition of network structures are located in the `./train` directory. The player code is in `./user_interface`, and the demo of the player is in `./demo`.
- The project uses Pianoroll format music data and trains a VQ-VAE model to learn the low-dimensional representation (latent variables) of music and an autoregressive generative model for these music latent variables.
<div align="center">
<img src="images/pianoroll.png" alt="Pianoroll music data" width="60%"/>
</div>

## Model Training
- Using an autoencoder model, we obtain an encoder and a decoder by compressing the music score into latent variables and reconstructing the score.
- Train an autoregressive model (PianoCNN3D) to generate the next piece of music score based on the latent variables of the previous piece.
<div align="center">
<img src="images/training.png" alt="Training schematic" width="60%"/>
</div>
- To enable the autoregressive model to predict the next segment of music latent variables in parallel during training, a masked 3D convolutional kernel (MASK A) is used.
<div align="center">
<img src="images/mask.png" alt="Masked 3D convolutional kernel" width="40%"/>
</div>

## Music Generation Player
- A music player developed using PyQt allows users to choose from 5 types of instrument combinations and randomly generate music with one click.
- The demo `demo/demo.mkv` demonstrates generating music, playing it in the player, and editing simple instrument tones.
- The `demo` further illustrates how the generated piano music can be exported as a MIDI file and edited in professional music software.
<div align="center">
<img src="images/image.png" alt="User interface" width="80%" />
</div>

### Running Instructions
Please install the Python dependencies listed in `./user_interface/requirements.txt` and run `python main.py` to start the player.

If you encounter the following error when using a conda environment:
```
OSError: .conda/envs/yourenv/lib/python3.10/site-packages/../../../././libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /usr/lib/libfluidsynth.so.3)
```
You can try resolving it by running `conda install -c conda-forge libstdcxx-ng`.
