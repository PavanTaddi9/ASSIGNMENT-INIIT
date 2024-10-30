import torch
import numpy as np
try:
    import torch_xla.core.xla_model as xm

    _xla_available = True
except ImportError:
    _xla_available = False

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.tpu:
                xm.master_print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )
            else:
                print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.tpu:
                xm.master_print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            else:
                print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            if self.tpu:
                xm.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

class CustomImageDataset(Dataset):
    def __init__(self, pixel_arrays, targets, resize=None, augmentations=None):
        self.pixel_arrays = pixel_arrays
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    def __len__(self):
        return len(self.pixel_arrays)
    def __getitem__(self, idx):
        image = np.array(self.pixel_arrays[idx], dtype=np.float32).reshape(48, 48)
        if self.resize:
            image = cv2.resize(image, self.resize)
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)  # Convert to RGB
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # For RGB, shape becomes (3, new_size, new_size)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return image, target
      
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
