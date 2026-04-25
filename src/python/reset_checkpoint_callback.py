import torch
from pathlib import Path

ckpt = torch.load("/home/zxcv/Downloads/lessac.ckpt", map_location="cpu")

# Reset callback val_loss
for cb_key, cb_state in ckpt["callbacks"].items():
    if "val_loss" in cb_key:
        cb_state["best_model_score"] = torch.tensor(float("inf"))
        cb_state["best_model_path"] = ""
        cb_state["current_score"] = None
        cb_state["best_k_models"] = {}
        cb_state["kth_best_model_path"] = ""
        cb_state["kth_value"] = torch.tensor(float("inf"))
        cb_state["last_model_path"] = ""

    # Reset callback monitor=None (save every epoch)
    if cb_state.get("monitor") is None:
        cb_state["best_model_path"] = ""
        cb_state["last_model_path"] = ""
        cb_state["dirpath"] = Path("/workspace/training/lightning_logs/version_0/checkpoints")

# Reset epoch/step
ckpt["epoch"] = 0
ckpt["global_step"] = 0

# Reset loops (PL dùng cái này để restore epoch/step khi resume)
if "loops" in ckpt:
    del ckpt["loops"]

torch.save(ckpt, "lessac_clean_base.ckpt")
print("Done")