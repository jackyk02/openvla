# DROID Dataset Training Setup for OpenVLA

This folder contains all necessary scripts and configurations for training OpenVLA on the DROID (Distributed Robot Interaction Dataset).

## 📁 Files Created

### Core Scripts
1. **`download_droid.sh`** - Download DROID dataset from Google Cloud Storage
2. **`train_droid_lora.sh`** - Quick-start LoRA fine-tuning script
3. **`train_droid_full.sh`** - Full training from scratch script
4. **`augment_droid_dataset.py`** - Extract and preprocess DROID dataset to JSON format
5. **`extract_droid_example.sh`** - Example script for dataset extraction
6. **`convert_droid_to_bridge.py`** - Convert DROID actions to Bridge scale via tokens
7. **`convert_droid_example.sh`** - Example script for action conversion
8. **`compute_droid_episode_scores.py`** - Compute VLA-CLIP scores for episode quality
9. **`compute_droid_scores_example.sh`** - Example script for score computation

### Documentation
- **`DROID_TRAINING_GUIDE.md`** - Complete guide with all instructions, troubleshooting, and examples
- **`ACTION_CONVERSION_README.md`** - Guide for converting DROID actions to Bridge scale
- **`DROID_VLA_CLIP_SCORING_README.md`** - Guide for computing episode quality scores

### Configuration Files (Modified)
- **`prismatic/vla/datasets/rlds/oxe/mixtures.py`** - Added DROID dataset mixture
- **`prismatic/conf/vla.py`** - Added DROID training configuration

## 🚀 Quick Start

### 1. Download DROID Dataset

```bash
# For testing (2GB subset)
./download_droid.sh subset /root/tfrecords

# For full dataset (1.7TB)
./download_droid.sh full /root/tfrecords
```

### 2. Train OpenVLA on DROID

**Option A: LoRA Fine-tuning (Recommended)**
```bash
# Edit train_droid_lora.sh to set your paths
nano train_droid_lora.sh

# Run training
./train_droid_lora.sh
```

**Option B: Direct Command**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /root/tfrecords \
  --dataset_name droid_100 \
  --run_root_dir ./runs/droid \
  --adapter_tmp_dir ./runs/droid/adapters \
  --lora_rank 32 \
  --batch_size 16 \
  --learning_rate 5e-4 \
  --image_aug True
```

### 3. (Optional) Extract Dataset for Custom Processing

```bash
# Quick test extraction
./extract_droid_example.sh

# Full extraction
python augment_droid_dataset.py \
  --builder_dir /root/tfrecords/droid_100/1.0.0 \
  --output_path droid_extracted.json \
  --use_exterior_camera
```

### 4. (Optional) Convert Actions to Bridge Scale

If fine-tuning a Bridge-trained model, convert DROID actions to Bridge scale:

```bash
# Convert using example script
./convert_droid_example.sh

# Or manually
python convert_droid_to_bridge.py \
  --input droid_extracted.json \
  --output droid_bridge_converted.json
```

See `ACTION_CONVERSION_README.md` for details on the token-based conversion method.

### 5. (Optional) Compute VLA-CLIP Quality Scores

Evaluate episode quality using VLA-CLIP scores (requires VLA-CLIP checkpoint):

```bash
# Update checkpoint path in script, then run:
./compute_droid_scores_example.sh

# Or manually
python compute_droid_episode_scores.py \
  --merged_checkpoint /path/to/merged_checkpoint.pt \
  --droid_dataset droid_bridge_converted.json \
  --images_folder droid_extracted_images
```

See `DROID_VLA_CLIP_SCORING_README.md` for details on filtering high-quality episodes.

## 📚 Key Features

### Dataset Extraction (`augment_droid_dataset.py`)
- ✅ **Exterior camera** (exterior_image_1_left) extraction
- ✅ **Image resizing** to 256x256 (configurable)
- ✅ **Memory-efficient** deduplication of action histories
- ✅ **Temporal windows** around each timestep (configurable)
- ✅ **Image extraction** to JPG format (quality=95)
- ✅ **Instruction rephrasing** support
- ✅ **JSON output** for easy integration

### Action Conversion (`convert_droid_to_bridge.py`)
- ✅ **Token-based conversion** from DROID to Bridge scale
- ✅ **Automatic gripper binarization** (<0.5→0.0, ≥0.5→1.0)
- ✅ **Preserves padding** values
- ✅ **Statistics and verification** included
- ✅ **Metadata tracking** of conversion parameters

### Quality Scoring (`compute_droid_episode_scores.py`)
- ✅ **VLA-CLIP scoring** for episode quality assessment
- ✅ **Batch inference** for efficiency
- ✅ **Per-episode statistics** (mean, std, min, max, median)
- ✅ **Sample-level scores** for fine-grained filtering
- ✅ **JSON output** with detailed results

### Training Configurations
- ✅ **LoRA fine-tuning** for single GPU (27GB+ VRAM)
- ✅ **Full training** for multi-GPU setup (8x GPUs recommended)
- ✅ **DROID-specific** data transforms and configurations
- ✅ **Compatible** with OpenVLA's existing infrastructure

## 📖 Documentation

For complete documentation, see **`DROID_TRAINING_GUIDE.md`** which includes:
- Detailed installation instructions
- Hardware requirements
- Training options and configurations
- Troubleshooting guide
- DROID dataset information
- Citation information

## 🔧 Key Differences from Bridge Dataset

| Feature | Bridge V2 | DROID |
|---------|-----------|-------|
| **Camera** | Agent view (external) | Exterior camera (exterior_image_1_left) |
| **Action Space** | 7-DoF | 7-DoF (6D velocity + gripper) |
| **Action Format** | Absolute position | Velocity control |
| **Image Key** | `image_0` | `exterior_image_1_left` |
| **Gripper** | Direct | Inverted (1 - gripper) |
| **Split** | `val` | `train` |

## 🎯 Training Configurations Available

1. **`prism-dinosiglip-224px+mx-droid`** - Full DROID dataset training
   - 8 GPUs
   - Global batch size: 256
   - Shuffle buffer: 500K

2. **`droid_100` via finetune.py** - LoRA fine-tuning on subset
   - 1 GPU (16GB+ VRAM)
   - Batch size: 16
   - LoRA rank: 32

3. **`droid_wipe` via finetune.py** - Task-specific fine-tuning
   - 1 GPU
   - Pre-configured for wiping task

## 📊 DROID Dataset Info

- **Size**: ~1.7TB (full), ~2GB (droid_100)
- **Format**: RLDS (Robot Learning Dataset Standard)
- **Action Dim**: 7 (6D Cartesian velocity + 1D gripper)
- **Cameras**: Exterior (2x) + Wrist (1x)
- **Tasks**: Multi-task manipulation demonstrations

## 🐛 Troubleshooting

Common issues and solutions:

1. **gsutil download fails**: Destination directory must exist first (fixed in download script)
2. **OOM errors**: Reduce batch size, increase grad accumulation steps
3. **Dataset not found**: Check path structure matches `${data_root_dir}/droid_100/1.0.0/`
4. **Action dimension mismatch**: Ensure consistent episode processing

See `DROID_TRAINING_GUIDE.md` for more troubleshooting help.

## 📞 Support

- Check `DROID_TRAINING_GUIDE.md` for detailed documentation
- Check `ACTION_CONVERSION_README.md` for action scale conversion
- Check `DROID_VLA_CLIP_SCORING_README.md` for quality scoring
- OpenVLA Issues: https://github.com/openvla/openvla/issues
- DROID Dataset: https://droid-dataset.github.io

## 📄 Citations

```bibtex
@article{khazatsky2024droid,
  title={DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset},
  author={Khazatsky, Alexander and Pertsch, Karl and Nair, Suraj and ...},
  journal={arXiv preprint arXiv:2403.12945},
  year={2024}
}

@article{kim24openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and ...},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

---

**Ready to train? Start with the Quick Start section above! 🚀**

