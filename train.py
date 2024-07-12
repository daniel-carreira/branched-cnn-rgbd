import os
import sys
import time
import copy
import getopt
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from resources.dataset import CustomDataset
from resources.network import BranchedCNN

def compute_metrics(confusion_matrix):
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)

    return precision.mean(), recall.mean(), f1_score.mean()

def main(argv):
    arg_dataset_rgb = None
    arg_dataset_depth = None
    arg_outdir = None
    arg_epoch = None
    arg_learning_rate = None
    arg_batch = None
    arg_help = "{0} --rgb-data <DIR> --depth-data <DIR> --outdir <DIR> --epoch <INT> --lr <FLOAT> --batch <INT>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "", ["help", "rgb-data=", "depth-data=", "outdir=", "epoch=",
        "lr=", "batch="])
    except:
        print(arg_help)
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("--rgb-data"):
            arg_dataset_rgb = arg
        elif opt in ("--depth-data"):
            arg_dataset_depth = arg
        elif opt in ("--outdir"):
            arg_outdir = arg
        elif opt in ("--lr"):
            arg_learning_rate = float(arg)
        elif opt in ("--epoch"):
            arg_epoch = int(arg)
        elif opt in ("--batch"):
            arg_batch = int(arg)
        else:
            print(arg_help)
            sys.exit(2)

    # Check if all required arguments are provided
    if None in [arg_dataset_rgb, arg_dataset_depth, arg_outdir, arg_learning_rate, arg_epoch, arg_batch]:
        print("Error: Missing required argument(s).")
        print(arg_help)
        sys.exit(2)

    image_transforms = {
        "train": transforms.Compose([
            transforms.Resize((228, 228)),
            transforms.RandomCrop(size=224),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor(),
        ]),
        "valid": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

    train_dataset = CustomDataset(f"{arg_dataset_rgb}/train", f"{arg_dataset_depth}/train", transform=image_transforms['train'], normalized=False)
    val_dataset = CustomDataset(f"{arg_dataset_rgb}/valid", f"{arg_dataset_depth}/valid", transform=image_transforms['valid'], normalized=False)
    test_dataset = CustomDataset(f"{arg_dataset_rgb}/test", f"{arg_dataset_depth}/test", transform=image_transforms['test'], normalized=False)
    
    # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
    idx_to_class = {v: k for k, v in train_dataset.rgb_dataset.class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(train_dataset)
    valid_data_size = len(val_dataset)
    test_data_size = len(test_dataset)

    # Dataset Loader
    dataset_train_loader = DataLoader(train_dataset, batch_size=arg_batch, shuffle=True)
    dataset_valid_loader = DataLoader(val_dataset, batch_size=arg_batch, shuffle=True)
    dataset_test_loader = DataLoader(test_dataset, batch_size=arg_batch, shuffle=True)

    # Network
    model_ft = BranchedCNN(num_classes=num_classes)

    # Unfreeze All Layers
    for param in model_ft.parameters():
        param.requires_grad = True

    # Parameters to update
    params_to_update = []
    for param in model_ft.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    # Send the model to the processing unit
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Unique output directory
    counter = 0
    history_path = ""
    while True:
        log_path = f"{counter:04d}-epochs-{arg_epoch}-lr-{arg_learning_rate}-batch-{arg_batch}"
        history_path = os.path.join(arg_outdir, log_path)
        if not os.path.exists(history_path):
            break
        counter += 1

    if not os.path.exists(history_path):
        os.makedirs(history_path)

    optimizer_ft = optim.Adam(params_to_update, lr=arg_batch, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # Train
    history = []
    best_loss = 100000.0
    best_epoch = None
    best_model = None

    print("==========  TRAIN  ==========")
    with open(os.path.join(history_path, 'train.log'), 'a+') as the_file:
        the_file.write('==========  TRAIN  ==========\n')

    for epoch_i in range(arg_epoch):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch_i+1, arg_epoch))

        # Set to training mode
        model_ft.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for (inputs_rgb, inputs_depth, labels) in dataset_train_loader:

            inputs_rgb = inputs_rgb.to(device)
            inputs_depth = inputs_depth.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer_ft.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model_ft(inputs_rgb, inputs_depth)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer_ft.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs_rgb.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs_rgb.size(0)


        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model_ft.eval()

            # Validation loop
            for inputs_rgb, inputs_depth, labels in dataset_valid_loader:
                inputs_rgb = inputs_rgb.to(device)
                inputs_depth = inputs_depth.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model_ft(inputs_rgb, inputs_depth)

                # Compute loss
                loss = criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs_rgb.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs_rgb.size(0)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch_i
            best_model = copy.deepcopy(model_ft.state_dict())

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        log = "\tTraining: Loss - {:.4f}, Accuracy - {:.2f}%\n\tValidation: Loss - {:.4f}, Accuracy - {:.2f}%\n\tTime: {:.4f}s".format(avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start)
        print(log)
        with open(os.path.join(history_path, 'train.log'), 'a+') as the_file:
            the_file.write(f'Epoch {epoch_i+1}:\n{log}\n')

        # Save if the model has best accuracy till now
        if epoch_i == arg_epoch - 1:
            model_ft.load_state_dict(best_model)
            torch.save(model_ft, os.path.join(history_path, 'model.pt'))

            log = f"\nBest Model from Epoch {best_epoch+1}"
            print(log)
            with open(os.path.join(history_path, 'train.log'), 'a+') as the_file:
                the_file.write(log + "\n")

    history = np.array(history)

    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0,2)
    plt.savefig(os.path.join(history_path, 'loss_curve.png'))

    plt.clf()

    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.savefig(os.path.join(history_path, 'accuracy_curve.png'))

    plt.clf()

    test_acc = 0.0
    test_loss = 0.0

    confusion_matrix = torch.zeros(num_classes, num_classes)

    print("==========  TEST  ==========")
    with open(os.path.join(history_path, 'test.log'), 'a+') as the_file:
        the_file.write('==========  TEST  ==========\n')

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model_ft.eval()

        # Validation loop
        for inputs_rgb, inputs_depth, labels in dataset_test_loader:
            inputs_rgb = inputs_rgb.to(device)
            inputs_depth = inputs_depth.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model_ft(inputs_rgb, inputs_depth)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs_rgb.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs_rgb.size(0)

    cm = confusion_matrix.cpu().numpy()
    plt.imshow(cm, cmap='gray_r')
    plt.colorbar()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.0f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > (cm.max() / 1.5) else "black")

    tick_marks = np.arange(len(confusion_matrix))
    plt.xticks(tick_marks, idx_to_class.values(), rotation=45)
    plt.yticks(tick_marks, idx_to_class.values())

    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(history_path, 'confusion_matrix.png'), bbox_inches="tight")

    plt.clf()

    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    precision, recall, f1_score = compute_metrics(cm)

    indiv_acc = (confusion_matrix.diag()/confusion_matrix.sum(1)).numpy()

    print("Test: Loss - {:.4f}, Accuracy - {:.2f}%, Precision - {:.2f}%, Recall - {:.2f}%, F1-Score - {:.2f}%".format(avg_test_loss, avg_test_acc*100, precision*100, recall*100, f1_score*100))
    print("Test Per Class:")

    for key in idx_to_class:
        print(f"{idx_to_class[key]} - {round(indiv_acc[key] * 100, 2)}%")

    print()

    with open(os.path.join(history_path, 'test.log'), 'a+') as the_file:
        the_file.write("Test: Loss - {:.4f}, Accuracy - {:.2f}%, Precision - {:.2f}%, Recall - {:.2f}%, F1-Score - {:.2f}%\n\n".format(avg_test_loss, avg_test_acc*100, precision*100, recall*100, f1_score*100))
        the_file.write('==========  TEST PER CLASS  ==========\n')

        for key in idx_to_class:
            the_file.write(f"{idx_to_class[key]} - {round(indiv_acc[key] * 100, 2)}%\n")
    

if __name__ == "__main__":
    main(sys.argv)