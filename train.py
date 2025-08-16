import os
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PairedAudioDataset
from model import ScorePredictor, MosPredictor
import laion_clap


random.seed(1984)

def train_single_branch(device, score_type='quality'):
    # Paths
    DATA_DIR = "./AuditScore/dataset"
    TRAIN_META_PATH = os.path.join(DATA_DIR, 'train_meta.json')
    VALID_META_PATH = os.path.join(DATA_DIR, 'valid_meta.json')
    ORI_AUDIO_DIR = os.path.join(DATA_DIR, 'ori_wavs')
    TAR_AUDIO_DIR = os.path.join(DATA_DIR, 'tar_wavs')

    # Checkpoint dir
    CKPT_DIR = f'./ckpt/{score_type}/'
    os.makedirs(CKPT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{CKPT_DIR}/tensorboard/")

    # Load frozen CLAP
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt('ckpt/laion_clap/630k-audioset-best.pt')
    for param in model.parameters():
        param.requires_grad = False

    # Load branch-specific predictor
    combined_input_dim = 512 * 4
    quality_mlp = ScorePredictor(combined_input_dim) if score_type == 'quality' else None
    relevance_mlp = ScorePredictor(combined_input_dim) if score_type == 'relevance' else None
    faithfulness_mlp = ScorePredictor(combined_input_dim) if score_type == 'faithfulness' else None

    net = MosPredictor(model, upstream_output_dim=512,
                   quality_mlp=quality_mlp,
                   relevance_mlp=relevance_mlp,
                   faithfulness_mlp=faithfulness_mlp).to(device)

    train_dataset = PairedAudioDataset(TRAIN_META_PATH, ORI_AUDIO_DIR, TAR_AUDIO_DIR)
    valid_dataset = PairedAudioDataset(VALID_META_PATH, ORI_AUDIO_DIR, TAR_AUDIO_DIR)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=valid_dataset.collate_fn)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    # Training loop
    TRAINING_EPOCH = 100
    SAVE_EVERY = 5
    global_step = 0

    for epoch in range(1, TRAINING_EPOCH + 1):
        net.train()
        epoch_train_loss = 0.0
        for data in tqdm(train_loader, desc=f"[{score_type}] Epoch {epoch} Training", ncols=100):
            ori_wavs, ori_prompts, tar_wavs, tar_prompts, quality, relevance, faithfulness = data
            ori_wavs, tar_wavs = ori_wavs.squeeze(1).to(device), tar_wavs.squeeze(1).to(device)

            label = {
                'quality': quality,
                'relevance': relevance,
                'faithfulness': faithfulness
            }[score_type].to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred_all = net(ori_wavs, ori_prompts, tar_wavs, tar_prompts)
            pred = pred_all[{'quality': 0, 'relevance': 1, 'faithfulness': 2}[score_type]]
            
            
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()

            writer.add_scalar(f"{score_type}/train_step_loss", loss.item(), global_step)
            # writer.add_scalar(f"{score_type}/train_step_loss_weighted", weighted_loss.item(), global_step)
            global_step += 1
            epoch_train_loss += loss.item()

        net.eval()
        epoch_valid_loss = 0.0
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f"[{score_type}] Epoch {epoch} Validating", ncols=100):
                ori_wavs, ori_prompts, tar_wavs, tar_prompts, quality, relevance, faithfulness = data
                ori_wavs, tar_wavs = ori_wavs.squeeze(1).to(device), tar_wavs.squeeze(1).to(device)

                label = {
                    'quality': quality,
                    'relevance': relevance,
                    'faithfulness': faithfulness
                }[score_type].to(device).unsqueeze(1)

                pred_all = net(ori_wavs, ori_prompts, tar_wavs, tar_prompts)
                pred = pred_all[{'quality': 0, 'relevance': 1, 'faithfulness': 2}[score_type]]
                loss = criterion(pred, label)
                epoch_valid_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_valid_loss = epoch_valid_loss / len(valid_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{score_type}] Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        writer.add_scalar(f"{score_type}/epoch_train_loss", avg_train_loss, epoch)
        writer.add_scalar(f"{score_type}/epoch_valid_loss", avg_valid_loss, epoch)
        writer.add_scalar(f"{score_type}/lr", current_lr, epoch)

        scheduler.step(avg_valid_loss)

        if epoch % SAVE_EVERY == 0:
            if score_type == 'quality':
                torch.save(quality_mlp.state_dict(), os.path.join(CKPT_DIR, f"{score_type}_epoch_{epoch}.pt"))
            elif score_type == 'relevance':
                torch.save(relevance_mlp.state_dict(), os.path.join(CKPT_DIR, f"{score_type}_epoch_{epoch}.pt"))
            elif score_type == 'faithfulness':
                torch.save(faithfulness_mlp.state_dict(), os.path.join(CKPT_DIR, f"{score_type}_epoch_{epoch}.pt"))

    writer.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    
    for score_type in ['quality', 'relevance', 'faithfulness']:
        train_single_branch(device, score_type)
