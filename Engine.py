class Engine:
    @staticmethod
    def train(data_loader, model, optimizer, device,scheduler=None, accumulation_steps=1, fp16=False):
        model.train()
        losses = AverageMeter()
        scaler = torch.cuda.amp.GradScaler() if fp16 else None
        if accumulation_steps > 1:
            optimizer.zero_grad()
        
        tk0 = tqdm(data_loader, total=len(data_loader))
        for batch_idx, (images, targets) in enumerate(tk0):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                    scaler.scale(loss).backward()
                else:
                    outputs = model(images)
                    loss = torch.nn.CrossEntropyLoss()(outputs,targets)
                    loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer) if fp16 else optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device, use_tpu=False):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                images, targets = data  
                images = images.to(device)
                targets = targets.to(device)
                predictions, loss = model(images, targets)
                predictions = predictions.cpu()
                losses.update(loss.item(), images.size(0))
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        final_predictions = torch.cat(final_predictions).numpy()
        return final_predictions, losses.avg
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            for b_idx, data in enumerate(tk0):
                inputs, _ = data  # Unpack data
                inputs = inputs.to(device)
                predictions = model(inputs)  # Assume model returns only predictions
                final_predictions.append(predictions.cpu())
                tk0.set_postfix()
        return torch.cat(final_predictions).numpy()
