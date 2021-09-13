# Importing libraries and helper functions
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from helper_code import*
from transformer import*
from dataset import*
from evaluate_model import*


def train(ep, model, optimizer, train_loader, device, criterion):
  model.train()
  total_loss = 0
  n_entries = 0
  all_logits = []
  all_labels = []
  train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
  train_bar = tqdm(initial=0, leave=True, total=len(train_loader),
                     desc=train_desc.format(ep, 0), position=0)
  for i, batch in enumerate(train_loader):
    data,target = batch
    data = data.to(device)
    target = target.to(device)
    model.zero_grad()
    output = model(data)
    logits_sigmoid = torch.sigmoid(output)
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()
    batch_loss = loss.detach().cpu().numpy()
    total_loss += batch_loss
    bs = target.size(0)
    n_entries += bs
    all_logits.append(logits_sigmoid.detach().cpu())
    all_labels.append(target.cpu())
    train_bar.desc = train_desc.format(ep, total_loss / n_entries)
    train_bar.update(bs)
  train_bar.close()
  return total_loss / n_entries,torch.cat(all_labels),torch.cat(all_logits)

def validate(ep, model, valid_loader, device, criterion):
    model.eval()
    total_loss = 0
    n_entries = 0
    all_logits = []
    all_labels = []
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(valid_loader),
                    desc=eval_desc.format(ep, 0), position=0)
    for i, batch in enumerate(valid_loader):
        with torch.no_grad():
            data,target = batch
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            loss = criterion(logits,target)
            total_loss += loss.item()
            logits_sigmoid = torch.sigmoid(logits)
            all_logits.append(logits_sigmoid.detach().cpu())
            all_labels.append(target.cpu())
            bs = data.size(0)
            n_entries += bs
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(bs)
    eval_bar.close()
    return total_loss / n_entries,torch.cat(all_labels),torch.cat(all_logits)

def test(model, test_loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            data,target = batch
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            logits_sigmoid = torch.sigmoid(logits)
            all_logits.append(logits_sigmoid.detach().cpu())
            all_labels.append(target.cpu())
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)
    all_labels = all_labels.numpy()
    all_logits = all_logits.numpy()
    all_labels = all_labels.astype(np.int)
    for i in range(len(all_logits)):
        all_logits[i] = (all_logits[i] >= 0.5).astype(np.int)
    return all_labels,all_logits

def get_scored_labels_index(lbls):
    cnt = 0
    index = []
    for i in range(len(lbls)):
        if np.sum(lbls[i])==0:
            cnt = cnt + 1
        else :
            index.append(i)
    print("No. of recording which doesn't have any scored labels are ",cnt)
    return index

def plot_labels_hist(idx_list,name,classes_short,dest,lbls):
  count = np.zeros(len(classes_short),dtype=int)
  for x in tqdm(idx_list):
    label = lbls[x]
    for i in range(len(label)):
      count[i] = count[i] + label[i]
  plt.figure(figsize=(35,10))
  plt.bar(classes_short, count)
  for index, value in enumerate(count):
      plt.text(index,value,
              str(value),fontsize = 'xx-large',horizontalalignment ='center',fontweight='bold')
  plt.title(name+ " Distribution")
  plt.savefig(os.path.join(dest,name+ " Distribution.png"))

def plot_results(results,name,dest):
    for keys in results:
        plt.figure()
        plt.title(name+" "+keys)
        plt.plot(results[keys])
        plt.savefig(os.path.join(dest,name+" "+keys+".png"))
        plt.close()

def plot_test_f1(arr,classes_short,dest):
    plt.figure(figsize=(35,10))
    plt.bar(classes_short, arr)
    for index, value in enumerate(arr):
        plt.text(index,value,
              str(round(value,4)),fontsize = 'xx-large',horizontalalignment ='center',fontweight='bold')
    plt.title("F1 Scores")
    plt.savefig(os.path.join(dest,"Test_F1_Score.png"))

def training_code(data_directory, save_directory):
    print('Finding header and recording files...')
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for saving if it does not already exist.
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

    # Extracting classes and weights
    print("Extracting classes and loading weights")
    weights_file = 'weights.csv'
    sinus_rhythm = set(['426783006'])
    classes, weights = load_weights(weights_file)
    classes_short = ["AF","AFL","BBB","Brady","CLBBB/LBBB","RBBB/CRBBB","IAVB","IRBBB","LAD","LAnFB","LPR","LQRSV","LQT","NSIVCB","NSR","PAC/SVPB","PR","PRWP","PVC/VPB","QAb","RAD","SA","SB","STach","TAb","TInv"]
    lbls = load_labels(header_files, classes)

    # Checking how many recordings doesn't have any scored labels
    indices = get_scored_labels_index(lbls)


    # # Preprocessing and saving the data
    # print("Preprocessing and saving the data")
    # cnt = 0
    # for i in tqdm(range(len(recording_files))):
    #     if np.sum(lbls[i])==0:
    #         continue
    #     recording = load_recording(recording_files[i])
    #     header = load_header(header_files[i])
    #     fs = get_frequency(header)
    #     recording = get_transformed_data(recording,fs)
    #     flag = 1
    #     for j in range(len(recording)):
    #         if(np.isnan(recording[j]).any()):
    #             flag = 0
    #             break
    #     if flag:
    #         np.save(os.path.join("data_processed", str(cnt)), recording)
    #         np.save(os.path.join("labels_processed", str(cnt)), lbls[i])
    #         cnt = cnt + 1


    cnt = len(indices)
    print("There are a total of {:d} recording for Train Valid and Test".format(cnt))


    # Splitting into 70% 10% and 20%
    train_idx, rem_idx = train_test_split(indices, test_size=0.3,random_state=42)
    val_idx,test_idx = train_test_split(rem_idx, test_size = 0.66,random_state=42)

    #Plotting the Train Valid and Test distribution
    plot_labels_hist(train_idx,"Train",classes_short,save_directory,lbls)
    plot_labels_hist(val_idx,"Valid",classes_short,save_directory,lbls)
    plot_labels_hist(test_idx,"Test",classes_short,save_directory,lbls)

    # HyperParameters
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 100
    emsize = 256 # embedding dimension == d_model
    dim_feedforward = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout_trans = 0.3 # the dropout value
    dropout_fc = 0.3 # dropout value for feedforward output layers
    n_class = len(classes)


    # Defing model and other params
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer(emsize,nhead,dim_feedforward,nlayers,n_class,dropout_trans,dropout_fc)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(device)
    train_dataset = ECG_Dataset(train_idx,lbls,recording_files,header_files)
    valid_dataset = ECG_Dataset(val_idx,lbls,recording_files,header_files)
    test_dataset = ECG_Dataset(test_idx,lbls,recording_files,header_files)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle= True,num_workers=4)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle= False,num_workers=4)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle= False,num_workers=4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    # Training
    print("Starting the training Process")
    train_results = defaultdict(list)
    valid_results = defaultdict(list)
    best_val_f1_macro = 0
    for epoch in tqdm(range(num_epochs)):
        t_loss,t_labels,t_outputs= train(epoch+1,model,optimizer,train_dl,device,criterion)
        v_loss,v_labels,v_outputs = validate(epoch+1,model,valid_dl,device,criterion)
        t_labels = t_labels.numpy()
        t_outputs = t_outputs.numpy()
        v_labels = v_labels.numpy()
        v_outputs = v_outputs.numpy()
        t_labels = t_labels.astype(np.int)
        v_labels = v_labels.astype(np.int)
        for i in range(len(t_outputs)):
            t_outputs[i] = (t_outputs[i] >= 0.5).astype(np.int)
        for i in range(len(v_outputs)):
            v_outputs[i] = (v_outputs[i] >= 0.5).astype(np.int)
        t_acc = compute_accuracy(t_labels,t_outputs)
        v_acc = compute_accuracy(v_labels,v_outputs)
        t_macro,t_micro,_ = compute_f_measure(t_labels,t_outputs)
        v_macro,v_micro,_ = compute_f_measure(v_labels,v_outputs)
        cscore_train = compute_challenge_metric(weights,t_labels,t_outputs,classes,sinus_rhythm)
        cscore_val = compute_challenge_metric(weights,v_labels,v_outputs,classes,sinus_rhythm)
        train_results["Accuracy"].append(t_acc)
        train_results["F1(Macro)"].append(t_macro)
        train_results["F1(Micro)"].append(t_micro)
        train_results["Loss"].append(t_loss)
        train_results["Challenge_Score"].append(cscore_train)
        valid_results["Accuracy"].append(v_acc)
        valid_results["F1(Macro)"].append(v_macro)
        valid_results["F1(Micro)"].append(v_micro)
        valid_results["Loss"].append(v_loss)
        valid_results["Challenge_Score"].append(cscore_val)
        if v_macro>best_val_f1_macro:
            best_val_f1_macro = v_macro
            torch.save(model.state_dict(),os.path.join(save_directory,"state_dict_model.pt"))


    # Testing
    print("Testing the model on test dataset")
    model.load_state_dict(torch.load(os.path.join(save_directory,"state_dict_model.pt")))
    test_labels,test_outputs = test(model,test_dl,device)
    test_acc = compute_accuracy(test_labels,test_outputs)
    test_macro,test_micro,test_f1 = compute_f_measure(test_labels,test_outputs)
    test_cmc_score = compute_challenge_metric(weights,test_labels,test_outputs,classes,sinus_rhythm)
    print("Test Accuracy ",test_acc)
    print("Test F1 Macro",test_macro)
    print("Test F1 Micro",test_micro)
    print("Test Challenge Score",test_cmc_score)
    file = open(os.path.join(save_directory,"Test_Results.txt"),"w")
    file.write("Test Accuracy :- {:f} \n".format(test_acc))
    file.write("Test F1 Macro :- {:f} \n".format(test_macro))
    file.write("Test F1 Micro :- {:f} \n".format(test_micro))
    file.write("Test Challenge Score :- {:f} \n".format(test_cmc_score))
    file.close()
    np.save(os.path.join(save_directory,"Test_F1"),test_f1)


    #Plotting the Training and Testing Results
    plot_results(train_results,"Train",save_directory)
    plot_results(valid_results,"Valid",save_directory)
    plot_test_f1(test_f1,classes_short,save_directory)

        
        
  