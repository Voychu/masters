from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import pandas as pd


def gender_prediction_noXAI(dataset, model_class, learning_rate, random_state, number_of_epochs, batch_size, device='cuda'):


    labels = dataset.df['sex_binary'].values
    #indeces = dataset.indeces
    #labels = full_df.iloc[indeces]['sex_binary'].values

    rskf = RepeatedStratifiedKFold(n_splits=2,n_repeats=5,random_state=random_state)
    indices = range(len(dataset))

    pandas_csv = []

    for fold, (train_index, test_index) in enumerate(rskf.split(indices,labels)):

        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        model = model_class().to(device)
        loss_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        train_losses=[]

        print(f'FOLD no: {fold}');
        for epoch in range(number_of_epochs):
            print(f'EPOCH no: {epoch}')
            model.train()
            avg_loss_from_batches = 0

            for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):

                X = batch['image'].to(device)
                y = batch['sex_binary'].to(device)

                pred = model(X)
                loss = loss_criterion(pred,y)
                avg_loss_from_batches+=loss.item()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss_from_batches = avg_loss_from_batches / len(train_loader)
            train_losses.append(avg_loss_from_batches)
            print(avg_loss_from_batches)


            model.eval()
            avg_test_loss_from_batches = 0
            correct = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Evaluation {epoch}'):
                    X_test = batch['image'].to(device)
                    y_test = batch['sex_binary'].to(device)

                    pred = model(X_test)
                    loss = loss_criterion(pred,y_test)
                    avg_test_loss_from_batches+= loss.item()
                    correct += (pred.argmax(1) == y_test).type(torch.float).sum().item()

            avg_test_loss_from_batches = avg_test_loss_from_batches / len(test_loader)
            accuracy = (correct / len(test_loader.dataset))*100

            print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_test_loss_from_batches:>8f} \n")

            pandas_csv.append({"Fold" : fold, "Epoch" : epoch, 
                               "Train loss" : avg_loss_from_batches, 
                               "Test loss" : avg_test_loss_from_batches,
                                "Accuracy" : accuracy})

    df = pd.DataFrame(pandas_csv)
    df.to_csv(f"{OUTPUT_DIR}/{model_class.__name__}_gender_noXAI.csv",index=False)
    print("Zakończono eksperyment")
