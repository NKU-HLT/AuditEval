import torch
import torchaudio
import os
import laion_clap
import warnings
import sys
import contextlib
from contextlib import contextmanager

from model import MosPredictor, ScorePredictor


warnings.filterwarnings("ignore")


@contextmanager
def enable_print():
    old_stdout = sys.stdout
    try:
        sys.stdout = old_stdout
        yield
    finally:
        sys.stdout = old_stdout


def load_audio(path, role="", target_length=160000):
    if role:
        print(f"[INFO] Loading {role} audio: {path}")
    else:
        print(f"[INFO] Loading audio: {path}")
    wav, sr = torchaudio.load(path)
    if wav.size(1) > target_length:
        wav = wav[:, :target_length]
    elif wav.size(1) < target_length:
        wav = torch.nn.functional.pad(wav, (0, target_length - wav.size(1)))
    return wav


def load_model(ckpt_paths, device):
    assert set(ckpt_paths.keys()) == {"quality", "relevance", "faithfulness"}

    print("[INFO] Loading CLAP upstream model: ./AuditEval-ssl/ckpt/laion_clap/630k-audioset-best.pt")
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        upstream_model = laion_clap.CLAP_Module(enable_fusion=False)
        upstream_model.load_ckpt('./AuditEval-ssl/ckpt/laion_clap/630k-audioset-best.pt')
    for param in upstream_model.parameters():
        param.requires_grad = False

    input_dim = 512 * 4

    print(f"[INFO] Loading checkpoint: {ckpt_paths['quality']}")
    quality_mlp = ScorePredictor(input_dim).to(device)
    quality_mlp.load_state_dict(torch.load(ckpt_paths["quality"], map_location=device))

    print(f"[INFO] Loading checkpoint: {ckpt_paths['relevance']}")
    relevance_mlp = ScorePredictor(input_dim).to(device)
    relevance_mlp.load_state_dict(torch.load(ckpt_paths["relevance"], map_location=device))

    print(f"[INFO] Loading checkpoint: {ckpt_paths['faithfulness']}\n")
    faithfulness_mlp = ScorePredictor(input_dim).to(device)
    faithfulness_mlp.load_state_dict(torch.load(ckpt_paths["faithfulness"], map_location=device))

    model = MosPredictor(
        upstream_model,
        upstream_output_dim=512,
        quality_mlp=quality_mlp,
        relevance_mlp=relevance_mlp,
        faithfulness_mlp=faithfulness_mlp
    ).to(device)
    model.eval()
    return model


def evaluate_single_sample(model, device, ori_path, tar_path, ori_text, tar_text):
    print("========================================== AuditEval-ssl ==========================================")

    ori_wav = load_audio(ori_path, role="Original").to(device)[0]
    print(f"[INFO] Original text: {ori_text}")
    print(f"[INFO] Target text:   {tar_text}\n")
    tar_wav = load_audio(tar_path, role="Target").to(device)[0]

    if ori_wav.shape[0] == 2:
        ori_wav = ori_wav[0:1, :]
    if tar_wav.shape[0] == 2:
        tar_wav = tar_wav[0:1, :]

    ori_wav = ori_wav.unsqueeze(0)
    tar_wav = tar_wav.unsqueeze(0)

    with torch.no_grad():
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            q_score, r_score, f_score = model(ori_wav, [ori_text], tar_wav, [tar_text])
        q_score = q_score.item()
        r_score = r_score.item()
        f_score = f_score.item()

    print("\n[RESULT] Quality Score:      {:.4f}".format(q_score))
    print("[RESULT] Relevance Score:    {:.4f}".format(r_score))
    print("[RESULT] Faithfulness Score: {:.4f}".format(f_score))
    print("========================================== AuditEval-ssl ==========================================\n")


if __name__ == "__main__":
    ckpt_paths = {
        "quality": "./AuditEval-ssl/ckpt/quality_epoch_50.pt",
        "relevance": "./AuditEval-ssl/ckpt/relevance_epoch_30.pt",
        "faithfulness": "./AuditEval-ssl/ckpt/faithfulness_epoch_35.pt"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_paths, device)

    ori_path = "./examples/sys1_add_ori_--8puiAGLhs_30000_40000.wav"
    ori_text = "Keys jingle as a car attempts to start."
    tar_text = "Keys jingle as a car attempts to start with a man speaking."

    # Example 1
    print("Example_1")
    tar_path = "./examples/sys1_add_tar_--8puiAGLhs_30000_40000.wav"
    evaluate_single_sample(model, device, ori_path, tar_path, ori_text, tar_text)

    # Example 2
    print("Example_2")
    tar_path = "./examples/sys2_add_tar_--8puiAGLhs_30000_40000.wav"
    evaluate_single_sample(model, device, ori_path, tar_path, ori_text, tar_text)

    # Example 3
    print("Example_3")
    tar_path = "./examples/sys3_add_tar_--8puiAGLhs_30000_40000.wav"
    evaluate_single_sample(model, device, ori_path, tar_path, ori_text, tar_text)
