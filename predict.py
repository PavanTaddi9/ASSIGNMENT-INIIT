def predict(fold,model_type,model_name):
    df = pd.read_csv("/kaggle/input/emo-map-challenge/test_dataset.csv")
    device = "cuda"  
    model_path = f"model_fold_{model_name}_{fold}.bin"  
    test_pixels = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.float32))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose([
        albumentations.Resize(height=224, width=224),  
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)  
    ])
    targets = np.zeros(len(df))
    test_dataset = CustomImageDataset(
        pixel_arrays=test_pixels,
        targets=targets,
        resize=None,  
        augmentations=aug,
    )
  
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    model = model_type 
    model.load_state_dict(torch.load(model_path))  
    model.to(device)  

    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions))  

    return predictions
