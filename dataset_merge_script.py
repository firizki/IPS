from bs4 import BeautifulSoup

def generate_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)

def generate_label(obj):
    return obj.find('name').text

def generate_target(file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        result = []
        for i in objects:
            result.append(soup.find('filename').text+","+generate_box(i)+","+generate_label(i))
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # # Labels (In my case, I only one class: target class or background)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        # # Tensorise img_id
        # img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        
        return result
    
# print(generate_target("datasets/andrewmvd_FaceMaskDataset/annotations/maksssksksss0.xml"))
import os

# Specify the directory path
directory = "datasets/andrewmvd_FaceMaskDataset/annotations/"

# Get a list of all files in the directory
files = os.listdir(directory)

file_path = "datasets/merged_dataset/label.csv"
with open(file_path, "a") as file:
    for filename in files:
        results = generate_target("datasets/andrewmvd_FaceMaskDataset/annotations/"+filename)
        for res in results:
            file.write(res+"\n")