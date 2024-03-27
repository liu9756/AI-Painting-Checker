import os
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class RealAISDLDataset(Dataset):

    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []

        # Traverse the images in the file, and label the files starting with ‘AI_’ as AI,
        # and the remaining labels as HUMAN
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(0 if os.path.basename(root).startswith('AI_') else 1)  # 0 AI, 1 HUMAN

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Randomly select 10 samples to ensure data introduction
def show_random_images(dataset, num_images=10):
    if len(dataset) < num_images:
        num_images = len(dataset)  # Match number of images to achieve balance

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    indices = random.sample(range(len(dataset)), num_images)  # Random choice

    for i, ax in enumerate(axes):
        image, label = dataset[indices[i]]
        ax.imshow(image.permute(1, 2, 0))  # matplotlib
        ax.set_title(label)
        ax.axis('off')

    plt.show()


# Analyze the dataset to see the statistic counts
def analyze_dataset(dataset):
    ai_count = dataset.labels.count(0)
    total_images = len(dataset)
    human_count = total_images - ai_count

    print(f"Total images counts: {total_images}")
    print(f"Label 'AI' counts: {ai_count}")
    print(f"Label 'HUMAN' counts: {human_count}")


# Balance the counts of AI and HUMAN data
# According to the analysis of train dataset, we reduce AI counts
def balance_dataset_inplace(dataset):
    ai_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    human_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

    min_length = min(len(ai_indices), len(human_indices))

    if len(ai_indices) > min_length:
        ai_indices = random.sample(ai_indices, min_length)

    balanced_indices = ai_indices + human_indices
    random.shuffle(balanced_indices)

    # update the dataset
    dataset.images = [dataset.images[i] for i in balanced_indices]
    dataset.labels = [dataset.labels[i] for i in balanced_indices]


# Transform the graph to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
])

# dataset establish
train_dataset = RealAISDLDataset('Real_AI_SD_LD_Dataset/train', transform=transform)
test_dataset = RealAISDLDataset('Real_AI_SD_LD_Dataset/test', transform=transform)

# dataset loading
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Display dataset for 10 images with labels
# show_random_images(train_dataset)

# Display the statistic analysis
# Train
print("Train dataset statistic information：")
analyze_dataset(train_dataset)

# Test
print("\nTest dataset statistic information：")
analyze_dataset(test_dataset)

# balance the dataset
balance_dataset_inplace(train_dataset)

# Update the train dataset
print("\nTrain dataset statistic information(After balance)：")
analyze_dataset(train_dataset)

# Total count
print("\n")
print(f"Balanced Training Dataset Size: {len(train_dataset)}")
print(f"Balanced Testing Dataset Size: {len(test_dataset)}")
