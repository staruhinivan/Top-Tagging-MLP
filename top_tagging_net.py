import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import LogLocator, NullFormatter
from activation_functions import PolPeriodic
from model import Model, find_peaks


# ====================== ЗАГРУЗКА ДАННЫХ ======================
print("Загружаем датасет...")

dataset_folder = "Few_dataset_cut" 

train_imp = torch.load(f"./{dataset_folder}/train_impulses.pt")   # shape: (N, ...)
train_tags = torch.load(f"./{dataset_folder}/train_tag.pt")   # shape: (N,)

val_imp = torch.load(f"./{dataset_folder}/val_impulses.pt")
val_tags = torch.load(f"./{dataset_folder}/val_tag.pt")

print(f"Загружено train: {train_imp.shape}, labels: {train_tags.shape}")
# print(f"Загружено test:  {test_imp.shape},  labels: {test_tags.shape}")
print(f"Загружено val:  {val_imp.shape},  labels: {val_tags.shape}")
print("Классы: 0 = isn'n top quark | 1 = top quark\n")

# ====================== ПАРАМЕТРЫ ======================
N_features = train_imp.shape[1]
trainable_param=False
vary_parab = False # many differ periodes are threated
neurons_destrib = False
Epoches_destrib=[0, 100, 200]
N = 1              # how many times the learning to repead, for statistic
learning_rate = 1e-3
weight_decay = 1e-1
l_decay = 0.1
num_epoches = 5                                                ### <----
neurons_lay=100 # number of neurons in each layer in the MLP   ### <----
data_size = 500 # train_imp.shape[0]                           ### <----
val_size = 100 # val_imp.shape[0]                              ### <----
batch_size = 10 #256                                           ### <----

# Split all learning on parts for saving memory
load_model = False # Boolean. Whether to load the model, that was saved before
save_model = True  # Boolean. Whether to save the model

split_fun = 0 # it is parameter to split all output on two files
# validation_split = 0.2
# Per_SawTooth=1
# Per_Parab=4
# Scale_Chebysh=0.5

# ======= ACTIVATION FUNCTIONS ========
# === list of the periodes of the Periodic Activation function (PolPeriodic) ===
periods=[0.4] # The Model will learn with ALL activation functions corresponding to these periods
names=[("period " + str(i)) for i in periods] # "Names" of the activation functions 
activations =[]  # Activation functions list

# === For Learing with the Periodic Act. function
for i in range(len(periods)):
    activations.append(PolPeriodic(per_init=periods[i], trainable=trainable_param))
# ======================

# ======= ПОДГОТОВКА ДАННЫХ =======

if len(train_imp) > data_size:
    train_imp = train_imp[:data_size]
    train_tags = train_tags[:data_size]
else:
    data_size=len(train_imp)

train_indices = list(range(data_size))
np.random.shuffle(train_indices)

# # Разбиваем на train / val
# split = int(np.floor(validation_split * data_size))
# train_indices = indices[split:]
# val_indices   = indices[:split]

## use validation

if len(val_imp) > val_size:
    val_imp = val_imp[:val_size]
    val_tags=val_tags[:val_size]
else:
    val_size=len(val_imp)

val_indices = list(range(val_size))
np.random.shuffle(val_indices)
##

## use test
# test_size = 10000
# if len(test_imp) > test_size:
#     test_imp = test_imp[:test_size]
#     test_tags = test_tags[:test_size]
# else:
#     data_size=len(train_imp)
# test_indices = list(range(test_size))
# np.random.shuffle(test_indices)
##

train_sampler = SubsetRandomSampler(train_indices)
val_sampler   = SubsetRandomSampler(val_indices)
# test_sampler = SubsetRandomSampler(test_indices)


# Создаём TensorDataset
train_dataset = TensorDataset(train_imp, train_tags)
val_dataset   = TensorDataset(train_imp, train_tags)   # используем тот же, но другой sampler
# test_dataset = TensorDataset(test_imp, test_tags)

# DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, sampler=val_sampler)
# test_loader  = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

# =================================================

# === Trainable periods ===
if trainable_param:
    history_PolPeriodic=np.zeros((N, num_epoches))
    history_Sawtooth=np.zeros((N, num_epoches))
    history_Chebyshev=np.zeros((N, num_epoches))


def train_model(model, train_loader, val_loader, loss, optimizer,  num_epochs, 
                optimizer_lr=None, neurons=None, save_period=False, iteration=None):    
    loss_history = []
    train_history = []
    val_history = [] 
    
    # ==== neurons destribution ====
    Neurons_all=[]
    

    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0

        if neurons is not None:
            layer1_outputs = []
            layer2_outputs = []
         

        # === neurons destribution ===        
        Neurons=[]

        for i_step, (x, y) in enumerate(train_loader):

            prediction = model(x)

            if neurons is not None:
                x1 = model[0](x)  # First Linear
                x1_=x1.detach()
                x1_activated = model[1](x1)  # First act
                x2 = model[2](x1_activated)  # Second Linear
                x2_=x2.detach()
                x2_activated = model[3](x2)  # Second act

                layer1_outputs.append(np.mean(x1_.numpy(), axis=0))
                layer2_outputs.append(np.mean(x2_.numpy(), axis=0))

            if (i_step<=10000) and (epoch in Epoches_destrib) and (neurons_destrib == True):
                x1 = model[0](x)  # First Linear
                x1_ = x1.detach()           

                list=(x1_.numpy()).copy()
                list = list.reshape(1, -1)[0]
                Neurons.append(list)
                ## i checked the hypothesis
                # list=np.sort(list, axis=1)
                # dif = np.abs(list[:, 1:]-list[:, :-1])
                # dif = dif.reshape(1,-1)                
                # Dif.append(dif[0])
            ##

            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y)
            total_samples += y.shape[0]
            
            loss_accum += loss_value           

        ##
        if neurons is not None:
            layer1_outputs=np.array(layer1_outputs)
            neurons['layer1'].append(np.mean(layer1_outputs, axis=0))
            layer2_outputs=np.array(layer2_outputs)
            neurons['layer2'].append(np.mean(layer2_outputs, axis=0))
        ##
        
        if save_period and (iteration is not None):
            for name, module in model.named_modules():             
                if hasattr(module[1], "T_parab"):
                    history_PolPeriodic[iteration, epoch]=(module[1].T_parab.item())
                if hasattr(module[1], "T_saw"):
                    history_Sawtooth[iteration, epoch]=(module[1].T_saw.item())
                if hasattr(module[1], "Scale_Chebysh"):
                    history_Chebyshev[iteration, epoch]=(module[1].Scale_Chebysh.item())
                break
                
        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        ### === learning rate optimization ===
        if(optimizer_lr):
            optimizer_lr.step(ave_loss)
        ### ===============================
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))
        #print_memory_info()

        # === neurons destribution === 
        #Dif_mean_hystory.append(np.mean(np.concatenate(Dif)))
        if (epoch in Epoches_destrib)  and (neurons_destrib == True):
            Neurons=np.concatenate(Neurons)
            Neurons_all.append(Neurons)

        # === check lr ===
        # current_lr = optimizer.param_groups[0]['lr']
        # check_lr.append(current_lr)

    return loss_history, train_history, val_history, Neurons_all #, Dif_mean_hystory
        
def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader
    
    Returns: accuracy as a float value between 0 and 1
    """
    model.eval() # Evaluation mode
    correct = 0
    total=0
    for x, y in loader:
        out = model(x)
        _, pred = torch.max(out, 1)
        total += y.shape[0]
        correct += torch.sum(pred == y)
    acc = float(correct)/total    
    return acc



# ======== Metrics =========
all_loss = np.zeros((len(activations)+1, N, num_epoches))
all_val_accuracy = np.zeros((len(activations)+1, N, num_epoches))
all_train_accuracy = np.zeros((len(activations)+1, N, num_epoches))


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

# === SAVE WEIGHTS ===
def save_model(model, optimizer, scheduler, i, iter):
    os.makedirs(f"./checkpoint_fun-{names[i]}_iter-{iter}", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),   # весь Adam
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, f"./checkpoint_fun-{names[i]}_iter-{iter}/model.pth")


def Network():
    for i, fun in enumerate(activations):
        for iter in range(N):        
            nn_model = Model(N_features, neurons_lay, fun)
            nn_model.type(torch.FloatTensor)
            #nn_model.apply(weights_init)

            # === LOAD MODEL ===
            if load_model == True:    
                checkpoint = torch.load(f"./checkpoint_fun-{names[i]}_iter-{iter}/model.pth")
                nn_model.load_state_dict(checkpoint["model_state_dict"])
                ##nn_model.eval()

            # ===== OPTIMIZER =====
            loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
            #optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

            # === LOAD OPTIMIZER===
            if load_model == True:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                #print("optimizer at the beginning:\n")
                #print(optimizer.param_groups[0]["lr"])

            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=l_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

            # === LOAD ===
            if load_model == True:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                #print("scheduler at the beginning:\n", scheduler.state_dict())

            #   if iter == 0:
            #     # neurons 
            #     # neurons = {'layer1': [], 'layer2': []}
                
            #     loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, num_epoches, optimizer_lr=scheduler)
            #     # layer1=np.array(neurons["layer1"])
            #     # layer2=np.array(neurons["layer2"])
            #     # plt.figure(figsize=(12,9))
            #     # plt.subplot(211)
            #     # plt.title(f'neurons layer1, {names[i]}')
            #     # for neuron in range(layer1.shape[1]):
            #     #   plt.plot(layer1[:, neuron], alpha=0.6, linewidth=1.5)
            #     # plt.ylabel('neurons')
            #     # plt.xlabel('epoches')
            #     # plt.subplot(212)
            #     # plt.title(f'neurons layer2, {names[i]}')
            #     # for neuron in range(layer2.shape[1]):
            #     #   plt.plot(layer2[:, neuron], alpha=0.6, linewidth=1.5)
            #     # plt.ylabel('neurons')
            #     # plt.xlabel('epoches')
            #     # plt.tight_layout()
            #     # plt.savefig(f"layers, {names[i]}")
            #     # plt.show()
            #   else:
            loss_history, train_history, val_history, neurons_all = train_model(nn_model, train_loader, val_loader, 
                        loss, optimizer, num_epoches, save_period=trainable_param, iteration=iter, optimizer_lr=scheduler) #test or val _loader!!!
            all_loss[i,iter] = loss_history
            all_val_accuracy[i, iter] = val_history
            all_train_accuracy[i, iter] = train_history

            # ====== SAVE MODEL ======
            if save_model == True:
                save_model(nn_model, optimizer, scheduler, i, iter)
            # print("optimizer at the end:\n")
            # print(optimizer.param_groups[0]["lr"])
            # print("scheduler at the end:\n", scheduler.state_dict())


            if iter == 0:
                plt.figure(figsize=(12,9))
                plt.subplot(311)
                plt.title(f"Loss with {names[i]}")
                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.plot(loss_history)
                plt.tight_layout()
                plt.grid(True)
                plt.subplot(312)
                plt.title(f"accuracy with {names[i]}")
                plt.ylabel("accuracy")
                plt.xlabel("epoch")
                plt.plot(val_history)
                plt.tight_layout()
                plt.grid(True)
                plt.subplot(313)
                plt.title(f"train accuracy {names[i]}")
                plt.plot(train_history)
                plt.tight_layout()
                plt.grid(True)
                plt.savefig(f'metrics, {names[i]}.png')

            if iter==0 and neurons_destrib==True:
                # === neurons destribution and spectrum ===
                for ep in range(len(Epoches_destrib)):
                    # ====================== ГИСТОГРАММА ======================
                    hist, bin_edges = np.histogram(neurons_all[ep], bins=1000)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    # ====================== FFT ======================
                    fft_values = np.fft.fft(hist)
                    fft_freq = np.fft.fftfreq(len(hist), d=bin_centers[1] - bin_centers[0])
                    magnitude = np.abs(fft_values)

                    # ====================== ПОИСК ЛОКАЛЬНЫХ МАКСИМУМОВ ======================
                    # Параметры можно подстраивать
                    peaks = find_peaks(magnitude, 
                                                height=np.max(magnitude)*0.001,   # минимальная высота пика (2% от самого высокого)
                                                distance=1,                     # минимальное расстояние между пиками (в точках)
                                                prominence=0.01)                # насколько пик "выделяется" от окружения

                    top_freq = fft_freq[peaks]
                    top_mag = magnitude[peaks]
                    mask = (top_freq <= 5 ) & (top_freq > 0)
                    top_freq = top_freq[mask]
                    top_mag = top_mag[mask]

                    # Сортируем по убыванию амплитуды
                    sort_idx = np.argsort(top_mag)[::-1]
                    top_freq = top_freq[sort_idx]
                    top_mag = top_mag[sort_idx]
                    top_n = 4  # сколько пиков показывать

                    # ====================== ГРАФИК ======================
                    plt.figure(figsize=(14, 6))

                    plt.subplot(1, 2, 1)
                    plt.hist(neurons_all[ep], bins=1000, density=True, alpha=0.75, color='skyblue', edgecolor='black')
                    plt.title('Гистограмма данных')
                    plt.xlabel('Значение')
                    plt.ylabel('Плотность вероятности')
                    plt.grid(True, alpha=0.3)

                    # ==================== СПЕКТР ====================
                    ax = plt.subplot(1, 2, 2)
                    plt.xlim([-10, 10])
                    plt.plot(fft_freq, magnitude, color='darkred', linewidth=1.1, alpha=0.7, label='Спектр')

                    colors = plt.cm.tab10(np.linspace(0, 1, top_n))  # красивые разные цвета

                    for i_, (f, m, c) in enumerate(zip(top_freq, top_mag, colors)):
                        # Точка    
                        ax.plot(f, m, 'o', color=c, markersize=5, markeredgecolor='black', markeredgewidth=0.5, 
                            label=f'{1./f:.3f}')
                                             
                    plt.title('Cпектр Фурье')
                    plt.xlabel('Частота')
                    plt.ylabel('Амплитуда')
                    plt.grid(True, alpha=0.3)
                    # Легенда с цветами
                    plt.legend(title='Период (Топ пики)', fontsize=9, title_fontsize=10, 
                            loc='upper right', bbox_to_anchor=(1.02, 1))

                    plt.tight_layout()
                    plt.savefig(f"neur_destr_per_{names[i]}_epoch_{Epoches_destrib[ep]}.png")
                    plt.close()



    # for iter in range(N):
    #   nn_model = nn.Sequential(
    #               Flattener(),
    #               nn.Linear(28*28, neurons_lay1),
    #               PolRand(seed = iter*10),
    #               nn.Linear(neurons_lay1, neurons_lay2), 
    #               PolRand(seed=iter*10),
    #               nn.Linear(neurons_lay2, 10)
    #             )
    #   nn_model.type(torch.FloatTensor)

    #   # We will minimize cross-entropy between the ground truth and
    #   # network predictions using an SGD optimizer
    #   loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
    #   optimizer = optim.SGD(nn_model.parameters(), lr=1e-2, weight_decay=1e-1)
    #   if iter == 0:
    #         # neurons 
    #         neurons = {'layer1': [], 'layer2': []}
            
    #         loss_history, train_history, val_history = train_model(nn_model, train_loader, val_loader, loss, optimizer, num_epoches, neurons=neurons)
    #         layer1=np.array(neurons["layer1"])
    #         layer2=np.array(neurons["layer2"])
    #         plt.figure(figsize=(12,9))
    #         plt.subplot(211)
    #         plt.title(f'neurons layer1, {names[i]}')
    #         for neuron in range(layer1.shape[1]):
    #           plt.plot(layer1[:, neuron], alpha=0.6, linewidth=1.5)
    #         plt.ylabel('neurons')
    #         plt.xlabel('epoches')
    #         plt.subplot(212)
    #         plt.title(f'neurons layer2, {names[i]}')
    #         for neuron in range(layer2.shape[1]):
    #           plt.plot(layer2[:, neuron], alpha=0.6, linewidth=1.5)
    #         plt.ylabel('neurons')
    #         plt.xlabel('epoches')
    #         plt.tight_layout()
    #         plt.savefig('layers Polinom')
    #         plt.show()
    #   else:
    #         loss_history, train_history, val_history = train_model(nn_model, train_loader, 
    #                                                                val_loader, loss, optimizer, num_epoches)

    #   all_loss[i,iter] = loss_history
    #   all_val_accuracy[i, iter] = val_history

    #   if iter == 0:
    #     plt.figure(figsize=(7,5))
    #     plt.subplot(411)
    #     plt.title(f"Loss with polinom")
    #     plt.ylabel("loss")
    #     plt.xlabel("epoch")
    #     plt.plot(loss_history)
    #     plt.subplot(412)
    #     plt.title(f"accuracy with polinom")
    #     plt.ylabel("accuracy")
    #     plt.xlabel("epoch")
    #     plt.plot(val_history)
    #     plt.subplot(413)
    #     plt.title(f"train accuracy polinom")
    #     plt.plot(train_history)      
    #     plt.tight_layout()
    #     plt.savefig('metrics polinom')
    #     plt.show()

Network()

# ========= VISUALIZATION =========
if N>1:
    all_loss=np.array(all_loss)
    x=np.arange(num_epoches)
    plt.figure(figsize=(12,9))
    for i in range(split_fun):
        y_mean = np.mean(all_loss[i], axis=0)
        y_std = np.std(all_loss[i], axis=0)
        y_min = np.min(all_loss[i], axis=0)
        y_max = np.max(all_loss[i], axis=0)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=names[i])
        #plt.plot(x, y_mean, label=names[i])

    # rand polinom
    # y_mean = np.mean(all_loss[i], axis=0)
    # y_std = np.std(all_loss[i], axis=0)
    # y_min = np.min(all_loss[i], axis=0)
    # y_max = np.max(all_loss[i], axis=0)
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Rand Polimnom')
    #plt.plot(x, y_mean, label='Rand polinom')

    plt.title("Losses for different activation functioins")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 150])
    plt.tight_layout()
    if split_fun>0:
        plt.savefig("Loss for different activation functions1.png")


    all_val_accuracy=np.array(all_val_accuracy)
    all_train_accuracy=np.array(all_train_accuracy)
    plt.figure(figsize=(12,9))
    plt.subplot(211)
    for i in range(split_fun):
        y_mean = np.mean(all_val_accuracy[i], axis=0)
        y_std = np.std(all_val_accuracy[i], axis=0)
        y_min = np.min(all_val_accuracy[i], axis=0)
        y_max = np.max(all_val_accuracy[i], axis=0)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=names[i])
        #plt.plot(x, y_mean, label=names[i])

    # rand polinom
    # y_mean = np.mean(all_val_accuracy[i], axis=0)
    # y_std = np.std(all_val_accuracy[i], axis=0)
    # y_min = np.min(all_val_accuracy[i], axis=0)
    # y_max = np.max(all_val_accuracy[i], axis=0)
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Rand Polimnom')
    #plt.plot(x, y_mean, label='Rand polinom')

    plt.title("Validation accuracy for different activation functioins")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 150])
    plt.tight_layout()

    plt.subplot(212)
    for i in range(split_fun):
        y_mean = np.mean(all_train_accuracy[i], axis=0)
        y_std = np.std(all_train_accuracy[i], axis=0)
        y_min = np.min(all_train_accuracy[i], axis=0)
        y_max = np.max(all_train_accuracy[i], axis=0)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=names[i])
        #plt.plot(x, y_mean, label=names[i])

    plt.title("Train accuracy for different activation functioins")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 150])
    plt.tight_layout()
    if split_fun>0:
        plt.savefig("Accuracy for different activation functions1.png")


    plt.figure(figsize=(12,9))
    for i in range(split_fun, len(activations)):
        y_mean = np.mean(all_loss[i], axis=0)
        y_std = np.std(all_loss[i], axis=0)
        y_min = np.min(all_loss[i], axis=0)
        y_max = np.max(all_loss[i], axis=0)

        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=names[i])
        #plt.plot(x, y_mean, label=names[i])

    # rand polinom
    # y_mean = np.mean(all_loss[i], axis=0)
    # y_std = np.std(all_loss[i], axis=0)
    # y_min = np.min(all_loss[i], axis=0)
    # y_max = np.max(all_loss[i], axis=0)
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Polimnom')
    #plt.plot(x, y_mean, label='polinom')

    plt.title("Losses for different activation functioins")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 150])
    plt.tight_layout()
    plt.savefig("Loss for different activation functions2.png")


    plt.figure(figsize=(12,9))
    plt.subplot(211)
    for i in range(split_fun, len(activations)):
        y_mean = np.mean(all_val_accuracy[i], axis=0)
        y_std = np.std(all_val_accuracy[i], axis=0)
        y_min = np.min(all_val_accuracy[i], axis=0)
        y_max = np.max(all_val_accuracy[i], axis=0)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=names[i])
        #plt.plot(x, y_mean, label=names[i])

    # rand polinom
    # y_mean = np.mean(all_val_accuracy[i], axis=0)
    # y_std = np.std(all_val_accuracy[i], axis=0)
    # y_min = np.min(all_val_accuracy[i], axis=0)
    # y_max = np.max(all_val_accuracy[i], axis=0)
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label='Polimnom')
    #plt.plot(x, y_mean, label='polinom')

    plt.title("Accuracy (validation) for different activation functions")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 150])
    plt.tight_layout()

    plt.subplot(212)
    for i in range(split_fun, len(activations)):
        y_mean = np.mean(all_train_accuracy[i], axis=0)
        y_std = np.std(all_train_accuracy[i], axis=0)
        y_min = np.min(all_train_accuracy[i], axis=0)
        y_max = np.max(all_train_accuracy[i], axis=0)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=names[i])
        #plt.plot(x, y_mean, label=names[i])

    plt.title("Train accuracy for different activation functions")
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    #plt.ylim([0, 150])
    plt.tight_layout()
    plt.savefig("Accuracy for different activation functions2.png")


    # plt.figure(figsize=(12,9))
    # plt.plot(x, rates)
    # plt.xlabel("epoches")
    # plt.ylabel("learning rate")
    # plt.title("learning rate")
    # plt.savefig("learning rate.png")

    ## Periods
    if trainable_param:
        plt.figure(figsize=(12,9))
        Cheb_min = np.min(history_Chebyshev, axis=0)
        Cheb_max = np.max(history_Chebyshev, axis=0)
        plt.fill_between(x, Cheb_min, Cheb_max, alpha=0.3, label='Chebyshev')
        Saw_min = np.min(history_Sawtooth, axis=0)
        Saw_max = np.max(history_Sawtooth, axis=0)
        plt.fill_between(x, Saw_min, Saw_max, alpha=0.3, label='SawTooth')
        Parab_min = np.min(history_PolPeriodic, axis=0)
        Parab_max = np.max(history_PolPeriodic, axis=0)
        plt.fill_between(x, Parab_min, Parab_max, alpha=0.3, label='Parabola Periodic')
        plt.legend()
        plt.title("Trainable period | scale")
        plt.xlabel("epoch")
        plt.ylabel("parameter")
        plt.grid(True)
        plt.tight_layout()    
        plt.savefig("Trainable periodes.png")


# === save different paraboles ===
if vary_parab:
    for i in range(len(names)):
        p = torch.tensor(all_loss[i, :], dtype=torch.float32)
        ac = torch.tensor(all_val_accuracy[i,:], dtype=torch.float32)
        #torch.save(p, f"./losses_parab_{names[i]}.pt")
        torch.save(ac, f"./val_acc_parab_{names[i]}.pt")


# === check lr ===
# plt.figure()
# plt.plot([i for i in range(num_epoches)], check_lr)
# plt.show()