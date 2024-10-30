def train(fold,model_type,model_name):
    
    df = pd.read_csv("/kaggle/input/train-data/train_folds.csv")
    device = "cuda"  
    epochs = 50  
    train_bs = 32  
    valid_bs = 16  

    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    
    train_pixels = df_train['pixels'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.float32))
    train_targets = df_train['emotion'].values
    valid_pixels = df_valid['pixels'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.float32))
    valid_targets = df_valid['emotion'].values

    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = A.Compose([
        A.Resize(height=224, width=224),  
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
        A.HorizontalFlip(p=0.5)  
    ])

   
    valid_aug = A.Compose([
        A.Resize(height=224, width=224), 
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True)
    ])

    
    train_dataset = CustomImageDataset(
        pixel_arrays=train_pixels,
        targets=train_targets,
        resize=(224, 224),  
        augmentations=train_aug,
    )
    valid_dataset = CustomImageDataset(
        pixel_arrays=valid_pixels,
        targets=valid_targets,
        resize=(224, 224), 
        augmentations=valid_aug,
    )

 
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )

  
    model = model_type
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5, 
        threshold=0.001, 
        mode="max"  
    )

    es = EarlyStopping(patience=5, mode="max")  

    for epoch in range(epochs):
        
        train_loss = Engine.train(train_loader, model, optimizer, device, scheduler=None, accumulation_steps=1, fp16=False)

        
        predictions, valid_loss = Engine.evaluate(valid_loader, model, device=device)
        final_predictions = np.argmax(np.vstack(predictions), axis=1)
        valid_targets = np.array(valid_targets)  
    
        
        accuracy = metrics.accuracy_score(valid_targets, final_predictions)
        print(f"Epoch = {epoch}, Accuracy = {accuracy:.4f}")

       
        scheduler.step(accuracy)
    
        
        es(accuracy, model, model_path=(f"model_fold_{model_name}_{fold}.bin"))
        if es.early_stop:
            print("Early stopping")
            break

   
    oof_data = {
        'id': df_valid.index,  
        'true_emotion': valid_targets,
        'pred_emotion': final_predictions,
    }
    
    return pd.DataFrame(oof_data)
