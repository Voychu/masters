from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score,roc_curve, roc_auc_score, confusion_matrix, mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from config import OUTPUT_DIR,MODEL_DIR


def gender_prediction_noXAI(dataset, model_class, learning_rate, random_state, number_of_epochs, batch_size, device='cuda'):


    labels = dataset.df['sex_binary'].values

    rskf = RepeatedStratifiedKFold(n_repeats=5,n_splits=2,random_state=random_state)
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

        best_accuracy = 0.0

        for epoch in range(number_of_epochs):
            print(f'EPOCH no: {epoch}')
            model.train()
            avg_loss_from_batches = 0

            for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):

                X = batch['image'].to(device)
                y = batch['gender'].to(device)

                optimizer.zero_grad()
                pred = model(X)
                loss = loss_criterion(pred,y)
                avg_loss_from_batches+=loss.item()
                
                loss.backward()
                optimizer.step()

            avg_loss_from_batches = avg_loss_from_batches / len(train_loader)
            train_losses.append(avg_loss_from_batches)
            print(avg_loss_from_batches)


            model.eval()
            avg_test_loss_from_batches = 0
            all_preds = []
            all_truths = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Evaluation {epoch}'):
                    X_test = batch['image'].to(device)
                    y_test = batch['gender'].to(device)

                    pred = model(X_test)
                    loss = loss_criterion(pred,y_test)
                    avg_test_loss_from_batches+= loss.item()
                    
                    all_preds.append(pred.argmax(1).cpu().numpy())
                    all_truths.append(y_test.cpu().numpy())

            avg_test_loss_from_batches = avg_test_loss_from_batches / len(test_loader)

            all_preds = np.concatenate(all_preds)
            all_truths = np.concatenate(all_truths)

            accuracy = accuracy_score(y_true=all_truths, y_pred=all_preds)
            f1s = f1_score(y_true=all_truths, y_pred=all_preds, zero_division=0)
            precision = precision_score(y_true=all_truths, y_pred=all_preds, zero_division=0)
            recall = recall_score(y_true=all_truths, y_pred=all_preds, zero_division=0)


            print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_test_loss_from_batches:>8f} \n")

            pandas_csv.append({"Fold" : fold, "Epoch" : epoch, 
                               "Train loss" : avg_loss_from_batches, 
                               "Test loss" : avg_test_loss_from_batches,
                                "Accuracy" : accuracy,
                                "F1 score" : f1s,
                                "precision" : precision,
                                "recall" : recall})
            if accuracy > best_accuracy:

                torch.save(model.state_dict(), f"{MODEL_DIR}/{model_class.__name__}_rstate_{random_state}_fold_{fold}_gender_best.pth")
                best_accuracy = accuracy

    df = pd.DataFrame(pandas_csv)
    df.to_csv(f"{OUTPUT_DIR}/{model_class.__name__}_gender_noXAI.csv",index=False)
    print("Zakończono eksperyment")



def age_prediction_noXAI(dataset,model_class,learning_rate,random_state,number_of_epochs,batch_size,device='cuda'):


    labels = dataset.df['age_bin_encoded'].values

    rskf = RepeatedStratifiedKFold(n_repeats=5,n_splits=2,random_state=random_state)
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

        best_accuracy =0.0

        for epoch in range(number_of_epochs):
            print(f'EPOCH no: {epoch}')
            model.train()
            avg_loss_from_batches = 0

            for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):

                optimizer.zero_grad()
                X = batch['image'].to(device)
                y = batch['age_bin'].to(device)

                pred = model(X)
                loss = loss_criterion(pred,y)
                avg_loss_from_batches+=loss.item()
                
                loss.backward()
                optimizer.step()

            avg_loss_from_batches = avg_loss_from_batches / len(train_loader)
            train_losses.append(avg_loss_from_batches)
            print(avg_loss_from_batches)


            model.eval()
            avg_test_loss_from_batches = 0
            all_preds = []
            all_truths = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Evaluation {epoch}'):
                    X_test = batch['image'].to(device)
                    y_test = batch['age_bin'].to(device)

                    pred = model(X_test)
                    loss = loss_criterion(pred,y_test)
                    avg_test_loss_from_batches+= loss.item()
                    
                    all_preds.append(pred.argmax(1).cpu().numpy())
                    all_truths.append(y_test.cpu().numpy())

            avg_test_loss_from_batches = avg_test_loss_from_batches / len(test_loader)

            all_preds = np.concatenate(all_preds)
            all_truths = np.concatenate(all_truths)

            accuracy = balanced_accuracy_score(y_true=all_truths, y_pred=all_preds)
            f1s = f1_score(y_true=all_truths, y_pred=all_preds, zero_division=0, average='weighted')
            precision = precision_score(y_true=all_truths, y_pred=all_preds, zero_division=0, average='weighted')
            recall = recall_score(y_true=all_truths, y_pred=all_preds, zero_division=0, average='weighted')


            print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_test_loss_from_batches:>8f} \n")

            pandas_csv.append({"Fold" : fold, "Epoch" : epoch, 
                               "Train loss" : avg_loss_from_batches, 
                               "Test loss" : avg_test_loss_from_batches,
                                "Accuracy" : accuracy,
                                "F1 score" : f1s,
                                "precision" : precision,
                                "recall" : recall})
            if accuracy > best_accuracy:

                torch.save(model.state_dict(), f"{MODEL_DIR}/{model_class.__name__}_rstate_{random_state}_fold_{fold}_gender_best.pth")
                best_accuracy = accuracy
    
    df = pd.DataFrame(pandas_csv)
    df.to_csv(f"{OUTPUT_DIR}/{model_class.__name__}_age_noXAI.csv",index=False)
    print("Zakończono eksperyment")


# def age_prediction_noXAI(dataset, model_class, learning_rate, random_state, number_of_epochs, batch_size, device='cuda'):


#     labels = dataset.df['age'].values

#     kf = KFold(n_splits=5,random_state=random_state)
#     indices = range(len(dataset))

#     pandas_csv = []

#     for fold, (train_index, test_index) in enumerate(kf.split(indices,labels)):

#         train_subset = Subset(dataset, train_index)
#         test_subset = Subset(dataset, test_index)
        
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)



#         model = model_class().to(device)
#         loss_criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#         train_losses=[]

#         print(f'FOLD no: {fold}');
        
#         for epoch in range(number_of_epochs):
#             print(f'EPOCH no: {epoch}')
#             model.train()
#             avg_loss_from_batches = 0

#             for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):

#                 X = batch['image'].to(device)
#                 y = batch['age'].to(device).float()
            
#                 pred = model(X).squeeze()
#                 loss = loss_criterion(pred,y)
#                 avg_loss_from_batches+=loss.item()
                
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#             avg_loss_from_batches = avg_loss_from_batches / len(train_loader)
#             train_losses.append(avg_loss_from_batches)
#             print(avg_loss_from_batches)


#             model.eval()
#             avg_test_loss_from_batches = 0
#             all_preds_t = []
#             all_truths_t = []
#             with torch.no_grad():
#                 for batch in tqdm(test_loader, desc=f'Evaluation {epoch}'):
#                     X_test = batch['image'].to(device)
#                     y_test = batch['age'].to(device).float()

#                     pred = model(X_test).squeeze()
#                     loss = loss_criterion(pred,y_test)
#                     avg_test_loss_from_batches+= loss.item()



#                     all_preds_t.append(pred.cpu().numpy())
#                     all_truths_t.append(y_test.cpu().numpy())


#             avg_test_loss_from_batches /= len(test_loader)
#             all_preds_t = np.concatenate(all_preds_t).ravel()
#             all_truths_t = np.concatenate(all_truths_t).ravel()

#             mae = mean_absolute_error(y_true=all_truths_t, y_pred=all_preds_t)
#             rmse = root_mean_squared_error(y_true=all_truths_t, y_pred=all_preds_t)

#             print(f"Test Error: \n MAE: {mae:>0.2f}, RMSE: {rmse:>0.2f}, Avg loss: {avg_test_loss_from_batches:>8f} \n")    
            
#             pandas_csv.append({"Fold" : fold, "Epoch" : epoch, 
#                                "Train loss" : avg_loss_from_batches, 
#                                "Test loss" : avg_test_loss_from_batches,
#                                "MAE" : mae,
#                                "RMSE" : rmse}
#                                )

#     df = pd.DataFrame(pandas_csv)
#     df.to_csv(f"{OUTPUT_DIR}/{model_class.__name__}_age_noXAI.csv",index=False)
#     print("Zakończono eksperyment")



# def multitask_noXAI(dataset, model_class, learning_rate, random_state, number_of_epochs, batch_size, device='cuda'):

#     labels = dataset.df['sex_binary'].values

#     skf = StratifiedKFold(n_splits=5,random_state=random_state)
#     indices = range(len(dataset))

#     pandas_csv = []

#     for fold, (train_index, test_index) in enumerate(skf.split(indices,labels)):

#         train_subset = Subset(dataset, train_index)
#         test_subset = Subset(dataset, test_index)
        
#         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

#         model = model_class().to(device)
#         loss_criterion_0 = torch.nn.CrossEntropyLoss()
#         loss_criterion_1 = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#         train_losses=[]

#         print(f'FOLD no: {fold}');
#         for epoch in range(number_of_epochs):
#             print(f'EPOCH no: {epoch}')
#             model.train()
#             avg_loss_from_batches = 0

#             for batch in tqdm(train_loader, desc=f'Train Epoch {epoch}'):

#                 X = batch['image'].to(device)
#                 y_0 = batch['sex_binary'].to(device)
#                 y_1 = batch['age'].to(device).float()
                


#                 pred_0, pred_1 = model(X)
#                 pred_1 = pred_1.squeeze()
#                 loss_0 = loss_criterion_0(pred_0,y_0)
#                 loss_1 = loss_criterion_1(pred_1,y_1)
#                 total_loss = loss_0+loss_1
#                 avg_loss_from_batches+= total_loss.item()

#                 total_loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#             avg_loss_from_batches = avg_loss_from_batches / len(train_loader)
#             train_losses.append(avg_loss_from_batches)
#             print(avg_loss_from_batches)


#             model.eval()
#             avg_test_loss_from_batches = 0
#             all_preds_0 = []
#             all_truths_0 = []
#             all_preds_1 = []
#             all_truths_1 = []
#             with torch.no_grad():
#                 for batch in tqdm(test_loader, desc=f'Evaluation {epoch}'):
#                     X_test = batch['image'].to(device)
#                     y_test_0 = batch['sex_binary'].to(device)
#                     y_test_1 = batch['age'].to(device).float()

#                     pred_0, pred_1 = model(X_test)
#                     pred_1 = pred_1.squeeze()
#                     loss_0 = loss_criterion_0(pred_0,y_test_0)
#                     loss_1 = loss_criterion_1(pred_1,y_test_1)
#                     total_loss = loss_0+loss_1
                    
#                     avg_test_loss_from_batches+= total_loss.item()

#                     all_preds_0.append(pred_0.argmax(1).cpu().numpy())
#                     all_truths_0.append(y_test_0.cpu().numpy())
                    
#                     all_preds_1.append(pred_1.cpu().numpy())
#                     all_truths_1.append(y_test_1.cpu().numpy())


         
#             avg_test_loss_from_batches /= len(test_loader)

#             all_preds_0 = np.concatenate(all_preds_0)
#             all_truths_0 = np.concatenate(all_truths_0)
#             all_preds_1 = np.concatenate(all_preds_1).ravel()
#             all_truths_1 = np.concatenate(all_truths_1).ravel()


#             accuracy = accuracy_score(y_true=all_truths_0, y_pred=all_preds_0)
#             f1s = f1_score(y_true=all_truths_0, y_pred=all_preds_0, zero_division=0)
#             precision = precision_score(y_true=all_truths_0, y_pred=all_preds_0, zero_division=0)
#             recall = recall_score(y_true=all_truths_0, y_pred=all_preds_0, zero_division=0)

#             mae = mean_absolute_error(y_true=all_truths_1, y_pred=all_preds_1)
#             rmse = root_mean_squared_error(y_true=all_truths_1, y_pred=all_preds_1)

#             print(f"Test Error: \n MAE: {mae:>0.2f}, RMSE: {rmse:>0.2f}, Avg loss: {avg_test_loss_from_batches:>8f} \n")    
            
#             pandas_csv.append({"Fold" : fold, "Epoch" : epoch, 
#                                "Train loss" : avg_loss_from_batches, 
#                                "Test loss" : avg_test_loss_from_batches,
#                                "MAE" : mae,
#                                "RMSE" : rmse,
#                                 "Accuracy" : accuracy,
#                                 "F1 score" : f1s,
#                                 "precision" : precision,
#                                 "recall" : recall}
#                                )

#     df = pd.DataFrame(pandas_csv)
#     df.to_csv(f"{OUTPUT_DIR}/{model_class.__name__}_mt_noXAI.csv",index=False)
#     print("Zakończono eksperyment")