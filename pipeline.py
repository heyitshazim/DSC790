import argparse
import math
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class WaveGANGenerator(nn.Module):
    def __init__(self, input_length: int, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=25, padding=12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=25, padding=12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, 1, kernel_size=25, padding=12),
            nn.Tanh(),
        )
        self.input_length = input_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaveGANDiscriminator(nn.Module):
    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=25, stride=4, padding=12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=25, stride=4, padding=12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=25, stride=4, padding=12),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMAutoregressive(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.tanh(self.head(out))


def load_audio(path: Path, sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


def preprocess_dataset(metadata_path: Path, audio_root: Path, subset: str, sr: int, max_files: int) -> list[np.ndarray]:
    metadata = pd.read_csv(metadata_path)
    subset_df = metadata[metadata["subset"] == subset]
    audio_samples = []
    for _, row in tqdm(subset_df.head(max_files).iterrows(), total=min(len(subset_df), max_files), desc=f"Loading {subset}"):
        track_id = int(row["track_id"])
        file_number = f"{track_id // 1000:03d}"
        file_id = f"{track_id:06d}"
        audio_path = audio_root / file_number / f"{file_id}.mp3"
        if audio_path.exists():
            audio_samples.append(load_audio(audio_path, sr))
    return audio_samples


def train_wavegan(audio_samples: list[np.ndarray], device: torch.device, epochs: int, batch_size: int) -> tuple[WaveGANGenerator, WaveGANDiscriminator]:
    if not audio_samples:
        raise ValueError("No audio samples provided for WaveGAN training.")
    max_len = max(len(sample) for sample in audio_samples)
    padded = [np.pad(sample, (0, max_len - len(sample))) for sample in audio_samples]
    data = torch.from_numpy(np.stack(padded)).unsqueeze(1).to(device)

    generator = WaveGANGenerator(input_length=max_len).to(device)
    discriminator = WaveGANDiscriminator().to(device)
    g_opt = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.9))
    d_opt = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
    criterion = nn.BCEWithLogitsLoss()

    for _ in tqdm(range(epochs), desc="WaveGAN training"):
        perm = torch.randperm(data.size(0))
        for idx in range(0, data.size(0), batch_size):
            batch = data[perm[idx:idx + batch_size]]
            if batch.size(0) == 0:
                continue
            noise = torch.randn_like(batch)
            fake_audio = generator(noise)

            d_opt.zero_grad()
            real_pred = discriminator(batch)
            fake_pred = discriminator(fake_audio.detach())
            real_loss = criterion(real_pred, torch.ones_like(real_pred))
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            fake_pred = discriminator(fake_audio)
            g_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            g_loss.backward()
            g_opt.step()

    return generator, discriminator


def train_lstm(audio_samples: list[np.ndarray], device: torch.device, epochs: int, batch_size: int, seq_len: int) -> LSTMAutoregressive:
    if not audio_samples:
        raise ValueError("No audio samples provided for LSTM training.")
    sequences = []
    for sample in audio_samples:
        if len(sample) < seq_len + 1:
            continue
        for idx in range(0, len(sample) - seq_len - 1, seq_len):
            sequences.append(sample[idx:idx + seq_len + 1])
    data = torch.from_numpy(np.stack(sequences)).float().to(device)

    model = LSTMAutoregressive().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for _ in tqdm(range(epochs), desc="LSTM training"):
        perm = torch.randperm(data.size(0))
        for idx in range(0, data.size(0), batch_size):
            batch = data[perm[idx:idx + batch_size]]
            if batch.size(0) == 0:
                continue
            inputs = batch[:, :-1].unsqueeze(-1)
            targets = batch[:, 1:].unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model


def compute_constraints(audio: np.ndarray, sr: int, window_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window_length = int(sr * window_ms / 1000)
    hop_length = window_length
    stft = librosa.stft(audio, n_fft=window_length, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_length)

    nu = np.maximum(freqs, 1e-6)
    th = 3.64 * (nu / 1000) ** (-0.8) - 6.5 * np.exp(-0.6 * (nu / 1000 - 3.3) ** 2)
    th = th[:, None]

    fft_mag = np.abs(np.fft.rfft(audio))
    fft_mag = np.maximum(fft_mag, 1e-12)
    mt = -20 * np.log10(fft_mag / 20e-6)

    return magnitude, th, mt, freqs


def create_constraints_per_frame(th: np.ndarray, mt: np.ndarray, frames: int, window_length: int, audio_len: int) -> tuple[np.ndarray, np.ndarray]:
    th_frame = np.repeat(th.mean(axis=0), frames)
    mt_frame = np.interp(np.linspace(0, len(mt) - 1, frames), np.arange(len(mt)), mt)
    th_samples = np.repeat(th_frame, window_length)[:audio_len]
    mt_samples = np.repeat(mt_frame, window_length)[:audio_len]
    return th_samples, mt_samples


def bilevel_optimization(audio: np.ndarray, models: dict[str, nn.Module], constraints: tuple[np.ndarray, np.ndarray],
                         device: torch.device, pgd_steps: int, inner_lr: float, outer_lr: float,
                         momentum: float) -> np.ndarray:
    audio_tensor = torch.from_numpy(audio).float().to(device)
    th_samples, mt_samples = constraints
    th_tensor = torch.from_numpy(th_samples).float().to(device)
    mt_tensor = torch.from_numpy(mt_samples).float().to(device)

    perturbations = []
    for model_name, model in models.items():
        delta = torch.zeros_like(audio_tensor, requires_grad=True)
        velocity = torch.zeros_like(audio_tensor)
        for _ in tqdm(range(pgd_steps), desc=f"Optimizing perturbation ({model_name})"):
            perturbed = torch.clamp(audio_tensor + delta, -1.0, 1.0)
            if isinstance(model, LSTMAutoregressive):
                output = model(perturbed.unsqueeze(0).unsqueeze(-1))
                loss = (output.squeeze() - perturbed).pow(2).mean()
            else:
                output = model(perturbed.unsqueeze(0).unsqueeze(0))
                loss = (output.squeeze() - perturbed).pow(2).mean()
            loss.backward()

            with torch.no_grad():
                delta -= inner_lr * delta.grad
                delta = torch.max(torch.min(delta, mt_tensor), th_tensor)
                delta.grad.zero_()
                velocity = momentum * velocity + delta
                delta += outer_lr * velocity
                delta = torch.max(torch.min(delta, mt_tensor), th_tensor)

        perturbations.append(delta.detach().cpu().numpy())

    return np.mean(np.stack(perturbations), axis=0)


def save_visualizations(output_dir: Path, audio: np.ndarray, magnitude: np.ndarray, freqs: np.ndarray, perturbation: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(output_dir / "waveform.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.imshow(20 * np.log10(magnitude + 1e-6), aspect="auto", origin="lower",
               extent=[0, magnitude.shape[1], freqs[0], freqs[-1]])
    plt.colorbar(label="dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(output_dir / "spectrogram.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(perturbation)
    plt.title("Perturbation")
    plt.tight_layout()
    plt.savefig(output_dir / "perturbation.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio dataset pipeline for surrogate model training and perturbation.")
    parser.add_argument("--audio-root", type=Path, required=True, help="Path to fma_full audio directory.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to tracks.csv metadata file.")
    parser.add_argument("--target", type=Path, required=True, help="Path to target audio file.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory for artifacts.")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--inner-lr", type=float, default=1e-2)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    small_samples = preprocess_dataset(args.metadata, args.audio_root, "small", args.sr, args.max_files)
    medium_samples = preprocess_dataset(args.metadata, args.audio_root, "medium", args.sr, args.max_files)
    medium_samples = small_samples + medium_samples

    print("Training surrogate models for small subset...")
    wavegan_small, _ = train_wavegan(small_samples, device, args.epochs, args.batch_size)
    lstm_small = train_lstm(small_samples, device, args.epochs, args.batch_size, seq_len=1024)

    print("Training surrogate models for medium subset...")
    wavegan_medium, _ = train_wavegan(medium_samples, device, args.epochs, args.batch_size)
    lstm_medium = train_lstm(medium_samples, device, args.epochs, args.batch_size, seq_len=1024)

    target_audio = load_audio(args.target, args.sr)
    magnitude, th, mt, freqs = compute_constraints(target_audio, args.sr)
    window_length = int(args.sr * 0.01)
    th_samples, mt_samples = create_constraints_per_frame(th, mt, magnitude.shape[1], window_length, len(target_audio))

    models = {
        "wavegan_small": wavegan_small,
        "lstm_small": lstm_small,
        "wavegan_medium": wavegan_medium,
        "lstm_medium": lstm_medium,
    }

    perturbation = bilevel_optimization(
        target_audio,
        models,
        (th_samples, mt_samples),
        device,
        args.pgd_steps,
        args.inner_lr,
        args.outer_lr,
        args.momentum,
    )

    perturbed_audio = np.clip(target_audio + perturbation, -1.0, 1.0)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_dir / f"{args.target.stem}_noise.wav"
    librosa.output.write_wav(output_name.as_posix(), perturbed_audio, sr=args.sr)

    save_visualizations(output_dir, target_audio, magnitude, freqs, perturbation)
    print(f"Saved noise-injected audio to {output_name}")


if __name__ == "__main__":
    main()
