import os, csv, json, random, torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
plt.rcParams["font.family"] = "Times New Roman"
os.environ["TORCH_CUDNN_V8_API_DISABLED"]= "1"
seed= 1234
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def create_imbalanced_indices(train_indices: np.array, all_labels: np.array, imbalance_ratio: dict, seed= 1234):
    random_generator = np.random.default_rng(seed)
    selected_indices= []
    for class_id in range(num_classes):
        class_indices= train_indices[all_labels[train_indices] == class_id]
        ratio= float(imbalance_ratio[class_id])
        keep_number= max(1, int(round(len(class_indices) * ratio)))
        retained_indices = random_generator.choice(class_indices, size=keep_number, replace=False)
        selected_indices.append(retained_indices)
    selected_indices= np.concatenate(selected_indices)
    random_generator.shuffle(selected_indices)
    return selected_indices.astype(np.int64)

### Create an imbalance dataset
def prepare_mnist_data(data_dir: str, validation_ratio: float, imbalance_ratio: dict, seed= 1234):
    MEAN, STD= 0.1307, 0.3081
    evaluation_transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(MEAN, ), std=(STD, ))])
    training_transform = transforms.Compose([transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.95, 1.05)), transforms.ToTensor(), transforms.Normalize(mean=(MEAN, ), std=(STD, ))])
    ### TRAIN DATASET
    training_dataset= datasets.MNIST(root=data_dir, train= True, download= True, transform= training_transform)
    test_dataset= datasets.MNIST(root=data_dir, train= False, download= True, transform=evaluation_transform)
    all_labels= np.asarray(training_dataset.targets, dtype= np.int64)
    all_indices= np.arange(len(all_labels))
    train_indices, validation_indices = train_test_split(all_indices, test_size= validation_ratio, random_state= seed, shuffle= True, stratify= all_labels)
    imbalanced_train_indices= create_imbalanced_indices(train_indices=train_indices, all_labels=all_labels, imbalance_ratio=imbalance_ratio, seed= seed)
    validation_dataset= datasets.MNIST(root=data_dir, train= True, download= True, transform= evaluation_transform)
    train_dataset= Subset(training_dataset, imbalanced_train_indices.tolist())
    validation_dataset= Subset(validation_dataset, validation_indices.tolist())
    train_labels = all_labels[imbalanced_train_indices]
    validation_labels = all_labels[validation_indices]
    test_labels = np.asarray(test_dataset.targets, dtype= np.int64)
    return train_dataset, validation_dataset, test_dataset, train_labels, validation_labels, test_labels

class CNN(nn.Module):
    def __init__(self, num_classes= 10, dropout_rate= 0.30):
        super().__init__()
        self.feature_extractor= nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x= self.feature_extractor(x)
        output= self.classifier(x)
        return output

def vis_confusion_matrix(ground_truth, predictions, num_classes= 10):
    cm= confusion_matrix(ground_truth, predictions, labels= np.arange(num_classes))
    row_sums= cm.sum(axis= 1, keepdims= True)
    cm_display= np.divide(cm.astype(float), row_sums, out= np.zeros_like(cm, dtype= float), where= row_sums != 0)
    value_format= ".2f"
    fig, ax= plt.subplots(figsize= (9, 8))
    confusion_matrix_image= ax.imshow(cm_display, interpolation= "nearest", cmap= "Blues")
    color_bar= fig.colorbar(confusion_matrix_image, ax= ax)
    color_bar.ax.tick_params(labelsize= 11)
    class_names= [str(class_id) for class_id in range(num_classes)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label", fontsize= 14)
    ax.set_ylabel("True Label", fontsize= 14)
    ax.set_title("Normalized Confusion Matrix on the Test Dataset", fontsize= 16)
    ax.tick_params(axis= "both", labelsize= 12)
    threshold= cm_display.max() / 2.0
    for true_label in range(cm_display.shape[0]):
        for predicted_label in range(cm_display.shape[1]):
            value= cm_display[true_label, predicted_label]
            ax.text(predicted_label, true_label, format(value, value_format), ha= "center", va= "center",color= ("white" if value > threshold else "black" ), fontsize= 11)
    plt.tight_layout()
    plt.show()
    plt.close()
    return cm

def prepare_mnist_gan_data(data_dir: str, validation_ratio: float, imbalance_ratio: dict, seed= 1234):
    ### GAN images must be normalized to [-1, 1]
    evaluation_transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean= (0.5,), std= (0.5,))])
    training_transform= transforms.Compose([
                                                                        transforms.RandomAffine(degrees= 8, translate= (0.08, 0.08), scale= (0.95, 1.05)),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean= (0.5,), std= (0.5,))])
    training_dataset= datasets.MNIST(root= data_dir, train= True, download= True, transform= training_transform)
    validation_dataset= datasets.MNIST(root= data_dir, train= True, download= True, transform= evaluation_transform)
    test_dataset= datasets.MNIST(root= data_dir,train= False, download= True, transform= evaluation_transform)
    all_labels= np.asarray(training_dataset.targets, dtype= np.int64)
    all_indices= np.arange(len(all_labels))
    train_indices, validation_indices= train_test_split(all_indices, test_size= validation_ratio, random_state= seed, shuffle= True, stratify= all_labels)
    imbalanced_train_indices= create_imbalanced_indices(train_indices= train_indices, all_labels= all_labels, imbalance_ratio= imbalance_ratio, seed= seed)
    train_dataset= Subset(training_dataset, imbalanced_train_indices.tolist())
    validation_dataset= Subset(validation_dataset, validation_indices.tolist())
    train_labels= all_labels[imbalanced_train_indices]
    validation_labels= all_labels[validation_indices]
    test_labels= np.asarray(test_dataset.targets, dtype= np.int64)
    return train_dataset, validation_dataset, test_dataset, train_labels, validation_labels, test_labels

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim= 100, num_classes= 10, label_embedding_dim= 32):
        super().__init__()
        self.label_embedding= nn.Embedding(num_embeddings= num_classes, embedding_dim= label_embedding_dim)
        self.input_layer= nn.Sequential(
                                                            nn.Linear(latent_dim + label_embedding_dim, 128 * 7 * 7),
                                                            nn.BatchNorm1d(128 * 7 * 7),
                                                            nn.ReLU(inplace= True)
                                                        )
        self.generator= nn.Sequential(
                                                        nn.Unflatten(dim= 1, unflattened_size= (128, 7, 7)),
                                                        nn.ConvTranspose2d(
                                                                                            in_channels= 128,
                                                                                            out_channels= 64,
                                                                                            kernel_size= 4,
                                                                                            stride= 2,
                                                                                            padding= 1
                                                                                        ),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(inplace= True),
                                                        nn.ConvTranspose2d(
                                                                                            in_channels= 64,
                                                                                            out_channels= 1,
                                                                                            kernel_size= 4,
                                                                                            stride= 2,
                                                                                            padding= 1
                                                                                        ),
                                                        nn.Tanh())
    def forward(self, noise, labels):
        label_features= self.label_embedding(labels)
        combined_input= torch.cat([noise, label_features], dim= 1)
        features= self.input_layer(combined_input)
        generated_images= self.generator(features)
        return generated_images

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes= 10):
        super().__init__()
        self.label_embedding= nn.Embedding(num_embeddings= num_classes, embedding_dim= 28 * 28)
        self.discriminator= nn.Sequential(
            nn.Conv2d(
                                in_channels= 2,
                                out_channels= 64,
                                kernel_size= 4,
                                stride= 2,
                                padding= 1
                            ),
            nn.LeakyReLU(
                                        negative_slope= 0.2,
                                        inplace= True
                                    ),
            nn.Dropout2d(0.30),
            nn.Conv2d(
                                in_channels= 64,
                                out_channels= 128,
                                kernel_size= 4,
                                stride= 2,
                                padding= 1
                            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(
                                        negative_slope= 0.2,
                                        inplace= True
                                    ),
            nn.Dropout2d(0.30),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
            )
    def forward(self, images, labels):
        label_images= self.label_embedding(labels)
        label_images= label_images.view( labels.size(0), 1, 28, 28)
        conditional_images= torch.cat([images, label_images],dim= 1)
        logits= self.discriminator(conditional_images)
        return logits

def initialize_gan_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, mean= 0.0, std= 0.02)
        if module.bias is not None: nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight.data, mean= 1.0, std= 0.02)
        nn.init.constant_(module.bias.data, 0.0)

class GeneratedMNISTDataset(Dataset):
    def __init__(self, generated_images: torch.Tensor, generated_labels: torch.Tensor):
        super().__init__()
        MEAN= 0.1307
        STD= 0.3081
        generated_images= generated_images.detach().cpu().float()
        generated_images= (generated_images + 1.0) / 2.0
        generated_images= torch.clamp( generated_images, min= 0.0, max= 1.0)
        self.images= (generated_images - MEAN) / STD
        self.labels= generated_labels.detach().cpu().long()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image= self.images[index]
        label= int(self.labels[index].item())
        return image, label

def plot_training_curves(train_loss_ls, validation_loss_ls, train_accuracy_ls, validation_accuracy_ls, validation_balanced_accuracy_ls, model_name= "my model"):
    history_lengths= [len(train_loss_ls), len(validation_loss_ls), len(train_accuracy_ls), len(validation_accuracy_ls), len(validation_balanced_accuracy_ls)]
    epochs= np.arange(1, len(train_loss_ls) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_ls, marker="o", linewidth=2, label="Training Loss")
    plt.plot(epochs, validation_loss_ls, marker="s", linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch", fontsize= 20)
    plt.ylabel("Loss", fontsize= 20)
    plt.title(f"{model_name} Training and Validation Loss Curves", fontsize=16)
    plt.xticks(epochs, fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.tick_params(axis= "both", labelsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy_ls, marker="o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, validation_accuracy_ls, marker="s", linewidth=2, label="Validation Accuracy")
    plt.plot(epochs, validation_balanced_accuracy_ls, marker="^", linewidth=2, label="Validation Balanced Accuracy")
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.title(f"{model_name} Training and Validation Accuracy Curves", fontsize= 20)
    plt.xticks(epochs, fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.ylim(0.0, 1.05)
    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True, linestyle="--", alpha= 0.4)
    plt.legend(fontsize= 18)
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__== "__main__":
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir= os.path.join(os.getcwd(), "../", "data")
    # save_dir= os.path.join(os.getcwd(), "mnist_cnn_gan_results")
    ### CNN hyperparameters
    num_classes= 10
    batch_size= 128
    cnn_epoch= 10
    cnn_learning_rate= 0.001
    cnn_weight_decay= 0.0001
    cnn_dropout_rate= 0.30
    validation_ratio= 0.10
    patience= 3
    use_augmentation= True
    ### GAN model hyperparameters
    gan_epoch= 100
    gan_batch_size= 128
    latent_dim= 100
    label_embedding_dim= 32
    gan_learning_rate= 0.0002
    gan_beta1= 0.5
    gan_beta2= 0.999
    generation_batch_size= 256
    ### None means balancing all classes to the largest real class.
    gan_target_class_number= None
    ### Simulate for the imbalance problem
    imbalance_ratio= {0: 1.0, 1: 1.0, 2: 0.8, 3: 0.8, 4: 0.6, 5: 0.6, 6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1}
    train_dataset, validation_dataset, test_dataset, train_labels, validation_labels, test_labels= prepare_mnist_data(data_dir= data_dir, validation_ratio= validation_ratio, imbalance_ratio= imbalance_ratio)
    ### Visualization for label imbalance distribution
    class_ids= np.arange(num_classes)
    train_counts = np.bincount(train_labels, minlength=num_classes)
    validation_counts = np.bincount(validation_labels, minlength=num_classes)
    test_counts= np.bincount(test_labels, minlength=num_classes)
    plt.figure(figsize=(12, 6))
    train_bars = plt.bar(class_ids- 0.25, train_counts, width= 0.25, label="Training dataset")
    validation_bars= plt.bar(class_ids, validation_counts, width= 0.25, label="Validation dataset")
    test_bars= plt.bar(class_ids+ 0.25, test_counts, width= 0.25, label="Test dataset")
    plt.xlabel("MNIST class", fontsize= 20)
    plt.ylabel("Number of samples", fontsize= 20)
    plt.title("Label Distribution of Training, Validation, and Test Datasets", fontsize= 20)
    plt.xticks(class_ids, [str(i) for i in class_ids], fontsize=20)
    plt.yticks(fontsize= 20)
    plt.legend(fontsize= 20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    #### Display the exact sample number above each bar.
    for bars in [train_bars, validation_bars, test_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/ 2, height+ 2, f"{int(height)}", ha= "center", va= "bottom", fontsize= 15, rotation= 75)
    plt.tight_layout()
    plt.show()
    ### TRAIN
    train_loader= DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    validation_loader= DataLoader(validation_dataset, batch_size= batch_size, shuffle= False)
    test_loader= DataLoader( test_dataset, batch_size= batch_size, shuffle= False)
    model= CNN(num_classes= 10, dropout_rate= 0.30).to(device)
    criterion= nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr= cnn_learning_rate, weight_decay= cnn_weight_decay)
    train_loss_ls, validation_loss_ls, train_accuracy_ls, validation_accuracy_ls, validation_balanced_accuracy_ls= [], [], [], [], []
    best_validation_loss= float("inf")
    early_stopping_counter= 0
    best_model_path= os.path.join(os.getcwd(), "model_weights", "best_model.pth")
    for e in range(1, cnn_epoch + 1):
        model.train(); total_loss= 0.0
        all_predictions, all_labels= [], []
        for images, labels in train_loader:
            images= images.to(device)
            labels= labels.to(device)
            optimizer.zero_grad()
            logits= model(images)
            loss= criterion(logits, labels)
            loss.backward()
            optimizer.step()
            predictions= torch.argmax(logits, dim= 1)
            total_loss+= loss.item() * images.size(0)
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        average_train_loss= total_loss / len(train_loader.dataset)
        train_accuracy= accuracy_score(all_labels, all_predictions)
        train_balanced_accuracy= balanced_accuracy_score(all_labels, all_predictions)
        ### VALIDATION
        model.eval()
        total_validation_loss= 0.0; validation_predictions, validation_ground_truth= [], []
        with torch.no_grad():
            for images, labels in validation_loader:
                images= images.to(device)
                labels= labels.to(device)
                logits= model(images)
                loss= criterion(logits, labels)
                predictions= torch.argmax(logits, dim= 1)
                total_validation_loss+= loss.item() * images.size(0)
                validation_predictions.extend(predictions.detach().cpu().numpy())
                validation_ground_truth.extend(labels.detach().cpu().numpy())
        average_validation_loss= (total_validation_loss / len(validation_loader.dataset))
        validation_accuracy= accuracy_score(validation_ground_truth, validation_predictions)
        validation_balanced_accuracy= balanced_accuracy_score(validation_ground_truth, validation_predictions)
        train_loss_ls.append(average_train_loss)
        validation_loss_ls.append(average_validation_loss)
        train_accuracy_ls.append(train_accuracy)
        validation_accuracy_ls.append(validation_accuracy)
        validation_balanced_accuracy_ls.append(validation_balanced_accuracy)
        print(f"Epoch [{e:02d}/{cnn_epoch:02d}] | "f"Train loss: {average_train_loss:.4f} | " f"Train acc: {train_accuracy:.4f} | " f"Train bala acc: {train_balanced_accuracy:.4f} | " f"Validation loss: {average_validation_loss:.4f} | " f"Validation acc: {validation_accuracy:.4f} | " f"Validation bala acc: {validation_balanced_accuracy:.4f}")
        ### SAVE THE BEST MODEL
        if average_validation_loss < best_validation_loss:
            best_validation_loss= average_validation_loss
            early_stopping_counter= 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stopping_counter+= 1
            print(f"Early stopping counter: " f"{early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print("Early stopping was triggered.")
                break
    ### TEST
    plot_training_curves(train_loss_ls, validation_loss_ls, train_accuracy_ls, validation_accuracy_ls, validation_balanced_accuracy_ls, model_name="CNN w/o cGAN")
    model.load_state_dict(torch.load(best_model_path, weights_only= True))
    total_test_loss= 0.0
    test_predictions, test_ground_truth= [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images= images.to(device)
            labels= labels.to(device)
            logits= model(images)
            loss= criterion(logits, labels)
            predictions= torch.argmax(logits, dim= 1)
            total_test_loss+= loss.item() * images.size(0)
            test_predictions.extend(predictions.detach().cpu().numpy())
            test_ground_truth.extend(labels.detach().cpu().numpy())
    average_test_loss= total_test_loss / len(test_loader.dataset)
    test_accuracy= accuracy_score(test_ground_truth, test_predictions)
    test_balanced_accuracy= balanced_accuracy_score(test_ground_truth, test_predictions)
    test_precision, test_recall, test_f1_score, _=  (precision_recall_fscore_support(test_ground_truth, test_predictions, average= "macro", zero_division= 0))
    print("\nTest Results: ")
    print(f"Test loss: {average_test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test balanced accuracy: {test_balanced_accuracy:.4f}")
    print(f"Test macro precision: {test_precision:.4f}")
    print(f"Test macro recall: {test_recall:.4f}")
    print(f"Test macro F1-score: {test_f1_score:.4f}")
    print("\nClassification Report")
    print(classification_report(test_ground_truth, test_predictions, digits= 2, zero_division= 0))
    print("\nConfusion Matrix")
    confusion_matrix(test_ground_truth, test_predictions)
    vis_confusion_matrix(ground_truth= test_ground_truth, predictions= test_predictions, num_classes= 10)
    ### GAN IMAGE
    gan_train_dataset, gan_validation_dataset, gan_test_dataset, gan_train_labels, gan_validation_labels, gan_test_labels= prepare_mnist_gan_data(data_dir= data_dir,validation_ratio= validation_ratio, imbalance_ratio= imbalance_ratio, seed= seed)
    gan_class_counts= np.bincount(gan_train_labels, minlength= num_classes)
    gan_class_weights= np.divide(1.0, gan_class_counts, out= np.zeros_like(gan_class_counts, dtype= np.float64), where= gan_class_counts != 0)
    gan_sample_weights= gan_class_weights[gan_train_labels]
    gan_sampler= WeightedRandomSampler(weights= torch.as_tensor(gan_sample_weights, dtype= torch.double), num_samples= len(gan_sample_weights),replacement= True)
    gan_train_loader= DataLoader(gan_train_dataset, batch_size= gan_batch_size, sampler= gan_sampler, shuffle= False,)
    class_ids= np.arange(num_classes)
    plt.figure(figsize= (10, 6))
    bars= plt.bar(class_ids, gan_class_counts)
    plt.xlabel("MNIST class", fontsize= 20)
    plt.ylabel("Number of Samples", fontsize= 20)
    plt.title( "Class Distribution of the GAN Training Dataset", fontsize= 20)
    plt.xticks(class_ids, fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.grid(axis= "y", linestyle= "--",alpha= 0.4)
    for bar, count in zip(bars, gan_class_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)), ha= "center", va= "bottom", fontsize= 18)
    plt.tight_layout()
    plt.show()
    ### TRAIN condition GAN
    generator= ConditionalGenerator(latent_dim= latent_dim,num_classes= num_classes, label_embedding_dim= label_embedding_dim).to(device)
    discriminator= ConditionalDiscriminator(num_classes= num_classes).to(device)
    generator.apply(initialize_gan_weights); discriminator.apply(initialize_gan_weights)
    gan_criterion= nn.BCEWithLogitsLoss()
    generator_optimizer= torch.optim.Adam(generator.parameters(), lr= gan_learning_rate, betas= (gan_beta1, gan_beta2))
    discriminator_optimizer= torch.optim.Adam(discriminator.parameters(),lr= gan_learning_rate, betas= (gan_beta1, gan_beta2))
    generator_loss_ls, discriminator_loss_ls= [], []
    noise_vector= torch.randn(num_classes, latent_dim, device= device)
    fixed_labels= torch.arange(num_classes, device= device,dtype= torch.long)
    for e in range(1, gan_epoch + 1):
        generator.train(); discriminator.train()
        total_generator_loss, total_discriminator_loss, total_samples= 0.0, 0.0, 0
        for real_images, real_labels in gan_train_loader:
            real_images= real_images.to(device, non_blocking= True)
            real_labels= real_labels.to(device, non_blocking= True)
            current_batch_size= real_images.size(0)
            real_targets= torch.ones(current_batch_size,1, device= device)
            fake_targets= torch.zeros(current_batch_size, 1, device= device)
            ### discriminator -  real image
            discriminator_optimizer.zero_grad(set_to_none= True)
            real_logits= discriminator(real_images, real_labels)
            discriminator_real_loss= gan_criterion(real_logits, real_targets)
            ### discriminator - fake image
            discriminator_noise= torch.randn(current_batch_size,latent_dim, device= device)
            discriminator_fake_labels= torch.randint(low= 0, high= num_classes, size= (current_batch_size,), device= device)
            fake_images= generator(discriminator_noise, discriminator_fake_labels)
            fake_logits= discriminator(fake_images.detach(), discriminator_fake_labels)
            discriminator_fake_loss= gan_criterion(fake_logits, fake_targets)
            discriminator_loss= (discriminator_real_loss+ discriminator_fake_loss) / 2.0
            discriminator_loss.backward()
            discriminator_optimizer.step()
            ### generator - real image
            generator_optimizer.zero_grad(set_to_none= True)
            generator_noise= torch.randn(current_batch_size, latent_dim, device= device)
            generator_labels= torch.randint(low= 0, high= num_classes, size= (current_batch_size,), device= device)
            generated_images= generator(generator_noise, generator_labels)
            generated_logits= discriminator(generated_images, generator_labels)
            generator_loss= gan_criterion(generated_logits, real_targets)
            generator_loss.backward()
            generator_optimizer.step()
            total_discriminator_loss+= (discriminator_loss.item()* current_batch_size)
            total_generator_loss+= (generator_loss.item()* current_batch_size)
            total_samples+= current_batch_size
        average_discriminator_loss= (total_discriminator_loss/ total_samples)
        average_generator_loss= (total_generator_loss/ total_samples)
        discriminator_loss_ls.append(average_discriminator_loss)
        generator_loss_ls.append(average_generator_loss)
        print(f"GAN Epoch [{e:03d}/{gan_epoch:03d}] | " f"Discriminator Loss: {average_discriminator_loss:.4f} | " f"Generator Loss: {average_generator_loss:.4f}")
        generator.eval()
        with torch.no_grad():
            fixed_generated_images= generator(noise_vector, fixed_labels)
        if e% 10== 0:
            images= (fixed_generated_images.detach().cpu()+ 1.0) / 2.0
            images= torch.clamp(images, min= 0.0, max= 1.0)
            fig, axes= plt.subplots(nrows= 2, ncols= 5, figsize= (12, 5))
            axes= axes.flatten()
            for image_idx in range(num_classes):
                image= images[image_idx, 0].numpy()
                axes[image_idx].imshow(image, cmap= "gray", vmin= 0.0, vmax= 1.0)
                axes[image_idx].set_title(f"Generated Label: {fixed_labels[image_idx].item()}", fontsize= 12)
                axes[image_idx].axis("off")
            plt.suptitle("Conditional GAN Generated MNIST Images", fontsize= 16)
            plt.tight_layout()
            plt.show()
            plt.close()
        checkpoint_path= os.path.join(os.getcwd(), "model_weights", "conditional_gan_checkpoint.pth")
        torch.save({"epoch": e, "generator_state_dict": generator.state_dict(), "discriminator_state_dict": discriminator.state_dict(), "generator_optimizer_state_dict": generator_optimizer.state_dict(), "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(), "generator_loss": average_generator_loss, "discriminator_loss": average_discriminator_loss}, checkpoint_path)
    ### GENERATE IMAGE TO SOLVE CLASS IMBALANCE
    generator= generator.to(device)
    generator.eval()
    ### COUNT THE REAL TRAINING SAMPLES
    real_class_counts= np.bincount(train_labels, minlength= num_classes)
    ### DETERMINE THE TARGET NUMBER FOR EACH CLASS
    if gan_target_class_number is None:
        target_class_number= int(real_class_counts.max())
    else:
        target_class_number= int(gan_target_class_number)
    print("\nReal training class distribution:")
    for class_id, class_count in enumerate(real_class_counts):
        print(f"Class {class_id}: {class_count} real samples")
    print(f"\nTarget number for each class: " f"{target_class_number}")
    number_to_generate= np.maximum(target_class_number - real_class_counts, 0)
    print("\nNumber of GAN samples to generate:")
    for class_id, generated_number in enumerate(number_to_generate):
        print(f"Class {class_id}: " f"{generated_number} generated samples")
    generated_image_batches, generated_label_batches= [], []
    with torch.no_grad():
        for class_id in range(num_classes):
            remaining_number= int(number_to_generate[class_id])
            while remaining_number > 0:
                current_generation_batch_size= min(generation_batch_size, remaining_number)
                noise= torch.randn(current_generation_batch_size, latent_dim, device= device)
                conditional_labels= torch.full(size= (current_generation_batch_size,), fill_value= class_id, dtype= torch.long, device= device)
                generated_images= generator(noise, conditional_labels)
                generated_image_batches.append(generated_images.detach().cpu())
                generated_label_batches.append(conditional_labels.detach().cpu())
                remaining_number-= current_generation_batch_size
    all_generated_images= torch.cat(generated_image_batches,dim= 0)
    all_generated_labels= torch.cat(generated_label_batches, dim= 0)
    print(f"\nTotal generated samples: " f"{len(all_generated_labels)}")
    print(f"Generated image shape: " f"{all_generated_images.shape}")

    generated_dataset= GeneratedMNISTDataset(generated_images= all_generated_images, generated_labels= all_generated_labels)
    combined_train_dataset= ConcatDataset([train_dataset, generated_dataset])
    combined_train_labels= np.concatenate([np.asarray(train_labels, dtype= np.int64), all_generated_labels.numpy().astype(np.int64)])
    combined_class_counts= np.bincount(combined_train_labels, minlength= num_classes)

    class_ids= np.arange(num_classes)
    bar_width= 0.35
    plt.figure(figsize= (11, 6))
    plt.bar(class_ids - bar_width / 2, real_class_counts, width= bar_width, label= "Original Training Dataset")
    plt.bar(class_ids + bar_width / 2, combined_class_counts, width= bar_width, label= "GAN-Augmented Training Dataset")
    plt.xlabel("MNIST Class", fontsize= 14)
    plt.ylabel("Number of Samples", fontsize= 14)
    plt.title("Class Distribution Before and After GAN Augmentation", fontsize= 16)
    plt.xticks(class_ids, fontsize= 12)
    plt.yticks(fontsize= 12)
    plt.grid(axis= "y", linestyle= "--",alpha= 0.4)
    plt.legend(fontsize= 12)
    plt.tight_layout()
    plt.show()

    final_cnn_model_path= os.path.join(os.getcwd(), "model_weights", "best_model.pth")
    combined_train_loader= DataLoader(combined_train_dataset, batch_size= batch_size, shuffle= True,)
    validation_loader= DataLoader(validation_dataset, batch_size= batch_size, shuffle= False,)
    test_loader= DataLoader(test_dataset, batch_size= batch_size, shuffle= False)
    final_cnn_model= CNN(num_classes= num_classes, dropout_rate= cnn_dropout_rate).to(device)
    final_cnn_criterion= nn.CrossEntropyLoss()
    final_cnn_optimizer= torch.optim.Adam(final_cnn_model.parameters(), lr= cnn_learning_rate, weight_decay= cnn_weight_decay)
    final_train_loss_ls, final_validation_loss_ls, final_train_accuracy_ls, final_validation_accuracy_ls= [], [], [], []
    final_train_balanced_accuracy_ls, final_validation_balanced_accuracy_ls= [], []
    best_validation_loss= float("inf")
    early_stopping_counter= 0
    for e in range(1, cnn_epoch + 1):
        final_cnn_model.train()
        total_train_loss= 0.0
        train_predictions, train_ground_truth= [], []
        for images, labels in combined_train_loader:
            images= images.to(device, non_blocking= True)
            labels= labels.to(device, non_blocking= True)
            final_cnn_optimizer.zero_grad(set_to_none= True)
            logits= final_cnn_model(images)
            loss= final_cnn_criterion(logits, labels)
            loss.backward()
            final_cnn_optimizer.step()
            predictions= torch.argmax(logits, dim= 1)

            total_train_loss+= (loss.item()* images.size(0))
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_ground_truth.extend(labels.detach().cpu().numpy())

        average_train_loss= (total_train_loss/ len(combined_train_loader.dataset))
        train_accuracy= accuracy_score(train_ground_truth, train_predictions)
        train_balanced_accuracy= balanced_accuracy_score(train_ground_truth, train_predictions)
        total_validation_loss= 0.0
        validation_predictions, validation_ground_truth= [], []
        with torch.no_grad():
            for images, labels in validation_loader:
                images= images.to(device, non_blocking= True)
                labels= labels.to(device, non_blocking= True)
                logits= final_cnn_model(images)
                loss= final_cnn_criterion(logits, labels)
                predictions= torch.argmax(logits, dim= 1)
                total_validation_loss+= (loss.item()* images.size(0))
                validation_predictions.extend(predictions.detach().cpu().numpy())
                validation_ground_truth.extend(labels.detach().cpu().numpy())
        average_validation_loss= (total_validation_loss/ len(validation_loader.dataset))
        validation_accuracy= accuracy_score(validation_ground_truth, validation_predictions)
        validation_balanced_accuracy= balanced_accuracy_score(validation_ground_truth, validation_predictions)
        final_train_loss_ls.append(average_train_loss)
        final_validation_loss_ls.append(average_validation_loss)
        final_train_accuracy_ls.append(train_accuracy)
        final_validation_accuracy_ls.append(validation_accuracy)
        final_train_balanced_accuracy_ls.append(train_balanced_accuracy)
        final_validation_balanced_accuracy_ls.append(validation_balanced_accuracy)
        print(f"Epoch [{e:02d}/{cnn_epoch:02d}] | " f"Train loss: {average_train_loss:.4f} | "f"Train acc: {train_accuracy:.4f} | " f"Train BACC: {train_balanced_accuracy:.4f} | " f"Validation loss: {average_validation_loss:.4f} | " f"Validation acc: {validation_accuracy:.4f} | " f"Validation bala acc: {validation_balanced_accuracy:.4f}")
        if average_validation_loss < best_validation_loss:
            best_validation_loss= average_validation_loss
            early_stopping_counter= 0
            torch.save(final_cnn_model.state_dict(), final_cnn_model_path)
        else:
            early_stopping_counter+= 1
            print(f"Early stopping counter: " f"{early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                print("Early stopping was triggered.")
                break
    plot_training_curves(final_train_loss_ls, final_validation_loss_ls, final_train_accuracy_ls, final_validation_accuracy_ls, final_train_balanced_accuracy_ls, model_name="CNN with cGAN")
    final_cnn_model.load_state_dict(torch.load(final_cnn_model_path, map_location= device, weights_only= True))
    final_cnn_model= final_cnn_model.to(device)
    final_cnn_model.eval()
    ### FINAL TEST
    total_test_loss= 0.0
    test_predictions, test_ground_truth= [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images= images.to(device, non_blocking= True)
            labels= labels.to(device, non_blocking= True)
            logits= final_cnn_model(images)
            loss= final_cnn_criterion(logits, labels)
            predictions= torch.argmax(logits, dim= 1)
            total_test_loss+= (loss.item() * images.size(0))
            test_predictions.extend(predictions.detach().cpu().numpy())
            test_ground_truth.extend(labels.detach().cpu().numpy())
    test_predictions= np.asarray(test_predictions, dtype= np.int64)
    test_ground_truth= np.asarray(test_ground_truth, dtype= np.int64)
    average_test_loss= (total_test_loss/ len(test_loader.dataset))
    test_accuracy= accuracy_score(test_ground_truth, test_predictions)
    test_balanced_accuracy= balanced_accuracy_score(test_ground_truth, test_predictions)
    test_precision, test_recall, test_f1_score, _= precision_recall_fscore_support(test_ground_truth, test_predictions, labels= np.arange(num_classes), average= "macro", zero_division= 0)
    print("Final GAN-Augmented CNN Test Results")
    print(f"Test Loss: "f"{average_test_loss:.4f}")
    print(f"Test Accuracy:"f"{test_accuracy:.4f}")
    print(f"Test Balanced Accuracy: "f"{test_balanced_accuracy:.4f}")
    print(f"Test Macro Precision:" f"{test_precision:.4f}")
    print(f"Test Macro Recall: "f"{test_recall:.4f}")
    print(f"Test Macro F1-score:"f"{test_f1_score:.4f}")
    vis_confusion_matrix(ground_truth= test_ground_truth, predictions= test_predictions, num_classes= 10)