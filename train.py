import time

def train(model, optimizer, criterion, train_dataloader, valid_dataloader, batch_size, n_epochs, device, patience, lr_decay):
    '''
    :param model: network model
    :param optimizer: ...
    :param criterion: loss function
    :param train_dataloader: ...
    :param valid_dataloader: ...
    :param batch_size: ...
    :param n_epochs: the total epoch number of training
    :param device: CPU or cuda
    :param patience: the tolerance of the valid datasets loss function, 
                     if patience = 10, the valid datasets loss values in the (n+1)-th epoch to the (n+10)-th epoch are greater than the n-th epoch, 
                     then stop early, stop the training and output
    :param lr_decay: the epoch number of learning rate update interval

    :return model: the trained model has been completed
    :return early_stop: training is (True) or no (False) early stop
    :return train_epochs_loss: a list for storing train datasets loss
    :return valid_epochs_loss: a list for storing valid datasets loss
    '''
    
    # Initialize parameters
    best_loss = float("inf")   # the best loss
    best_epoch = 0   # the epoch corresponding to the best loss
    early_stop = False   # whether to stop early
    
    train_epochs_loss = []
    valid_epochs_loss = []
    
    # The loop of training begins
    for epoch in range(n_epochs):
        print("-"*30, "Epoch", epoch+1, "/", n_epochs, "-"*30)
        
        # Learning rate update
        if epoch > 0 and epoch % lr_decay == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        # Print learning rate
        print("Learning rate =", optimizer.param_groups[0]['lr'])
        
        # Train model
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for idx, (data, target) in enumerate(train_dataloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)   # put tensor into the network to get the predicted value
            loss = criterion(output, target)   # calculate the difference between the predicted value and the corresponding label
            
            # optimizer
            optimizer.zero_grad()   # when updating the network parameters in each iteration, it is necessary to clear the previous gradient to 0, 
                                    # otherwise the previous gradient will accumulate to this time.
            loss.backward()   # back propagation
            optimizer.step()   # the optimizer performs the next iteration
            
            train_loss += loss.item()
            if idx % 30 == 0:
                print("Train epoch: {}/{} [{}/{} ({:.1f}%)]\tLoss: {:.8f}".format(epoch+1, n_epochs, idx*batch_size, 
                     len(train_dataloader.dataset), 100.*idx*batch_size/len(train_dataloader.dataset), loss.item()/len(data)))
        
        train_epochs_loss.append(train_loss/len(train_dataloader.dataset))
        print("Train Loss: {}".format(train_loss/len(train_dataloader.dataset)))
        
        # Valid model
        model.eval()
        valid_loss = 0.0
        for idx, (data, target) in enumerate(valid_dataloader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            valid_loss += loss.item()
        
        valid_epochs_loss.append(valid_loss/len(valid_dataloader.dataset))
        print("Valid Loss: {}".format(valid_loss/len(valid_dataloader.dataset)))
        
        # Print the time consumed for each epoch training and validation
        end_time = time.time()
        print("{:.0f}min {:.0f}sec".format((end_time-start_time)//60, (end_time-start_time)%60))
        print("\n")
        
        # Determine whether early stop is required
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print("Early stopping on epoch {}".format(epoch+1))
            early_stop = True
            break
            
    return model, early_stop, train_epochs_loss, valid_epochs_loss, epoch+1