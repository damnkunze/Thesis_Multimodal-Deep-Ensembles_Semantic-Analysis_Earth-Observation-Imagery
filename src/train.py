import torch.cuda

from src.base import *
from src.metrics import *
from src.show import *


# RUN TRAINING

def run_train(train_dataloader, val_dataloader, model, criterion, optimizer, lr_scheduler, num_classes, device=None):
    """ Train the passed model until converted, meaning so Val loss didnt improve over last 'early_stopping_patience' epochs.
    If MAX_EPOCHS is set, train max until then.
    
    Expects to be set:
    NUM_CLASSES
    MAX_EPOCHS
    EARLY_STOPPING_PATIENCE

    Run with eg
    run_train(train_dataloader, val_dataloader, model, criterion, optimizer, lr_scheduler)
    """
    
    if device is None:
        device = DEVICE
    
    torch.cuda.empty_cache()

    early_stopping = False
    early_stopping_counter = 0
    early_stopping_best_score = 1000 # normally first at 3 then 1
    early_stopping_best_state = None

    # Redefine MAX_EPOCHS locally because it cannot be changed imported
    max_epochs = MAX_EPOCHS

    # Without MAX_EPOCHS, use early_stopping
    if max_epochs is None:
        early_stopping = True
        print("RUNNING WITH EARLY STOPPING. PATIENCE:", EARLY_STOPPING_PATIENCE)
        max_epochs = 1000

    # record for plotting later
    train_start_time = datetime.now().strftime("%d_%m_%y-%H:%M:%S")
    train_losses = [] # loss per epoch
    train_dcs = []
    val_losses = []
    val_dcs = []
    lrs = []

    # EPOCHS
    for epoch in tqdm(range(max_epochs)):
        model.train()

        last_lr = optimizer.param_groups[0]["lr"]
        lrs.append(last_lr)
        print("curr LR:", last_lr)

        train_running_loss = 0
        train_running_dc = np.zeros(num_classes - 1) # dont record 0
        train_count_batches = 0
        train_count_images = 0

        # All images in dataset
        for data_batch in tqdm(train_dataloader, leave=True, desc="Going through TRAIN dataset"):
            # img_batch shape
            # |- RGB [BATCH_SIZE, 3, 512, 512]
            # |- RGB, HSI [BATCH_SIZE, 53, 512, 512]
            img_batch = data_batch[0].float().to(device)

            # CrossEntropyLoss expects raw classes (as long)
            # mask shape [BATCH_SIZE, 512, 512]
            mask_batch = data_batch[1].long().to(device)

            # -> call to model() runs forward() because of nn.Module superclass
            # Expects batched inputs
            # Does all imgs in batch concurrently, out shape: [BATCH_SIZE, NUM_CLASSES, SIZE, SIZE]
            y_pred_batch = model(img_batch)

            # clear gradients to not track old operations, same as model.zero_grad()
            optimizer.zero_grad()

            # takes mean of full loss [BATCH_SIZE, 512, 512]
            loss = criterion(y_pred_batch, mask_batch)
            train_running_loss += loss.item()

            if loss.item() > 4 or torch.isnan(torch.tensor(loss.item())):
                print("!!!\n!!!\nloss > 4 or NaN", loss.item())
                print(f"Mask batch unique values: {torch.unique(mask_batch)}")

            # TODO parallelize
            """ 
            batch_dice = dice_coefficient_multiclass_batch(y_pred_batch, mask_batch)  # [B, NUM_CLASSES-1]
            train_running_dc += batch_dice.sum(dim=0).detach().cpu().numpy()
            train_count_images += mask_batch.shape[0]
            """
            for y_pred, mask in zip(y_pred_batch, mask_batch):
                train_running_dc += dice_coefficient_multiclass(y_pred, mask, num_classes)
                train_count_images += 1

            loss.backward()

            # Clip Gradients against exploding gradients (nan losses), (seems to make convergence faster too)
            # There are around 4 images in trainset that cause loss to explode > 4, which cause nan loss long term
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM_FOR_CLIPPING)

            optimizer.step()
            train_count_batches += 1

            # Show every epoch's: last batch's: first img, mask, pred
            # TODO BROKEN
            """ show_epochs_last_batch_first_img(train_count_batches, len(dataloader) - 1, img_batch, mask_batch, y_pred_batch, dataloader.dataset.channels_per_modality) """

            # Show whole first batch: img, mask, pred
            # TODO BROKEN
            ''' show_whole_batch(img_batch, mask_batch, y_pred_batch)
            return '''
            
            """ # For Debugging: skip train
            break """


        train_loss = train_running_loss / train_count_batches
        train_dc = train_running_dc / train_count_images

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        # VAL: Cross validation ---
        model.eval()

        val_running_loss = 0
        val_running_dc = np.zeros(num_classes - 1)
        val_count_batches = 0
        val_count_images = 0

        with torch.no_grad():
            for data in tqdm(val_dataloader, position=0, leave=True, desc="Going through VAL dataset"):
                val_count_batches += 1

                img = data[0].float().to(device)
                mask_batch = data[1].long().to(device)

                y_pred_batch = model(img)
                loss = criterion(y_pred_batch, mask_batch)
                val_running_loss += loss.item()

                # TODO parallelize
                for y_pred, mask in zip(y_pred_batch, mask_batch):
                    val_running_dc += dice_coefficient_multiclass(y_pred, mask, num_classes)
                    val_count_images += 1


        val_loss = val_running_loss / val_count_batches
        val_dc = val_running_dc / val_count_images

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        # Update LR
        # pass val_loss which should be minimized
        lr_scheduler.step(val_loss)

        # Early Stopping
        if early_stopping:
            if early_stopping_best_score is None or val_loss < early_stopping_best_score:
                print(f"New best VAL loss: {val_loss} Saved model state.")
                early_stopping_best_score = val_loss
                early_stopping_best_state = model.state_dict()
                early_stopping_counter = 0

            else:
                early_stopping_counter += 1
                print(f"\nEARLY STOPPING: NO IMPROVEMENT SINCE {early_stopping_counter}/{EARLY_STOPPING_PATIENCE} EPOCHS (Best VAL loss: {early_stopping_best_score:.4f} at epoch {epoch + 1 - early_stopping_counter})")

            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Max epochs without improvement reached {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}. Loading best result from VAL Loss: {val_loss}")
                model.load_state_dict(early_stopping_best_state)
                break

        print("-" * 30)
        print(f"Training Loss after EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Validation Loss after EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

    if early_stopping is None:
        print(f"\nMAX_EPOCHS reached: {MAX_EPOCHS}/{MAX_EPOCHS}\n")


    train_stats = {
        "train_start_time": train_start_time,
        "train_end_time": datetime.now().strftime("%d_%m_%y-%H:%M:%S"),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE if early_stopping else None,
        "train_losses": train_losses,
        "train_dcs": train_dcs,
        "val_losses": val_losses,
        "val_dcs": val_dcs,
        "lrs": lrs
    }

    return train_stats

    """ return train_losses, train_dcs, val_losses, val_dcs, lrs """



