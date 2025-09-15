# Cell 1: Install Dependencies
# Install Python 3.10 and required packages (matches your provided cell)
!sudo apt-get update -y
!sudo apt-get install python3.10 python3.10-dev python3.10-distutils -y
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
!sudo update-alternatives --config python3
# Select Python 3.10 when prompted (usually option 1)
!python3 -V
!wget https://bootstrap.pypa.io/get-pip.py
!python3 get-pip.py
!pip install nibabel einops tensorboardX
!pip install monai==0.8.0 numpy==1.23.5

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Set Working Directory
import os
os.chdir('/content/drive/My Drive/DLPROJECT/UNETRMODEL')
!pwd  # Verify current directory

# Cell 4: Verify Dataset and Code
# Check if dataset and key files exist
!ls ../MSD/Task09_Spleen
!ls ../MSD/Task09_Spleen/imagesTr
!ls ../MSD/Task09_Spleen/labelsTr
!ls ../MSD/Task09_Spleen/imagesTs
!ls networks
!cat ../MSD/Task09_Spleen/dataset_subset.json

# Cell 5: Train Base UNETR
# Train the base UNETR model and save checkpoints
!python main.py \
  --model_name unetr \
  --data_dir ../MSD/Task09_Spleen \
  --json_list dataset_subset.json \
  --logdir unetr_base \
  --save_checkpoint \
  --max_epochs 100 \
  --val_every 10 \
  --batch_size 1 \
  --sw_batch_size 1 \
  --optim_lr 1e-4 \
  --lrschedule warmup_cosine \
  --infer_overlap 0.5 \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 16 \
  --workers 2  # Reduced for Colab stability

# Cell 6: Train Attention-Gated UNETR
# Train the attention-gated UNETR model and save checkpoints
!python main.py \
  --model_name unetr_attention \
  --data_dir ../MSD/Task09_Spleen \
  --json_list dataset_subset.json \
  --logdir unetr_attention \
  --save_checkpoint \
  --max_epochs 100 \
  --val_every 10 \
  --batch_size 1 \
  --sw_batch_size 1 \
  --optim_lr 1e-4 \
  --lrschedule warmup_cosine \
  --infer_overlap 0.5 \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 16 \
  --workers 2  # Reduced for Colab stability

# Cell 7: Plot Training Loss and Validation Dice
# Load TensorBoard logs and plot Training Loss and Validation Dice vs. Epochs
%load_ext tensorboard
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def load_tensorboard_logs(logdir, scalar_names=['train_loss', 'val_acc']):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    data = {name: [] for name in scalar_names}
    steps = {name: [] for name in scalar_names}
    for name in scalar_names:
        events = event_acc.Scalars(name)
        data[name] = [event.value for event in events]
        steps[name] = [event.step for event in events]
    return data, steps

# Load logs for both models
base_data, base_steps = load_tensorboard_logs('/content/drive/My Drive/DLPROJECT/UNETRMODEL/unetr_base')
attn_data, attn_steps = load_tensorboard_logs('/content/drive/My Drive/DLPROJECT/UNETRMODEL/unetr_attention')

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(base_steps['train_loss'], base_data['train_loss'], label='Base UNETR')
plt.plot(attn_steps['train_loss'], attn_data['train_loss'], label='Attention-Gated UNETR')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot Validation Dice
plt.figure(figsize=(10, 5))
plt.plot(base_steps['val_acc'], base_data['val_acc'], label='Base UNETR')
plt.plot(attn_steps['val_acc'], attn_data['val_acc'], label='Attention-Gated UNETR')
plt.xlabel('Epoch')
plt.ylabel('Validation Dice Score')
plt.title('Validation Dice Score vs. Epoch')
plt.legend()
plt.grid()
plt.show()

# Cell 8: Verify Saved Models
# Check if model checkpoints were saved
!ls unetr_base
!ls unetr_attention

# Cell 9: Test Base UNETR
# Run testing on the test set using the best base UNETR model
!python test.py \
  --data_dir ../MSD/Task09_Spleen \
  --json_list dataset_subset.json \
  --pretrained_dir unetr_base \
  --pretrained_model_name model.pt \
  --saved_checkpoint ckpt \
  --infer_overlap 0.5 \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 16 \
  --workers 2

# Cell 10: Test Attention-Gated UNETR
# Run testing on the test set using the best attention-gated UNETR model
!python test.py \
  --data_dir ../MSD/Task09_Spleen \
  --json_list dataset_subset.json \
  --pretrained_dir unetr_attention \
  --pretrained_model_name model.pt \
  --saved_checkpoint ckpt \
  --infer_overlap 0.5 \
  --in_channels 1 \
  --out_channels 2 \
  --feature_size 16 \
  --workers 2