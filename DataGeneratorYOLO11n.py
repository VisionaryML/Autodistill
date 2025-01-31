import os
import shutil
from groundingdinoeye import Inference
from ultralytics import SAM

grounding_dino_eye = Inference()
groundingdino_model = grounding_dino_eye.LoadModel()
model = SAM(model="mobile_sam.pt")
model.info()

TEXT_PROMPT = "round tab"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25
image_directory = r'D:\Autodistill\imgs'
DEVICE = 'cpu'


def convert_mask_to_yolo_format(masks, image_name, data_dir):
    """
    Converts mask annotations into YOLOv5 format and splits dataset into training and validation sets.

    :param masks: List of mask objects containing the .xyn attributes (class, x, y, normalized coordinates)
    :param output_dir: Directory to save the YOLO annotations
    :param img_dir: Directory containing image files
    :param train_ratio: Ratio of training images, the rest will be validation
    """
    # Ensure output directories exist
    train_img_dir = os.path.join(data_dir, 'train/images')
    train_label_dir = os.path.join(data_dir, 'train/labels')
    
    # val_img_dir = os.path.join(data_dir, 'val/images')
    # val_label_dir = os.path.join(data_dir, 'val/labels')
    
    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        # os.makedirs(val_img_dir, exist_ok=True)
        # os.makedirs(val_label_dir, exist_ok=True)
    
    # Save the annotation to a text file
    _, extension = os.path.splitext(image_name)
    yolo_annotation = []
    for mask in masks:
        yolo_annotation.append(f"{0} ") 
        for annotation in mask.xyn[0]:
            x, y= annotation
            yolo_annotation.append(f"{x} {y} ")
        yolo_annotation.append(f"\n")
    img_name = image_name.replace(extension, '.txt')  # Assuming image file name corresponds
    with open(os.path.join(train_label_dir, img_name), 'w') as f:
        f.writelines(yolo_annotation)
    
    # Copy the image to the correct folder (train or val)
    shutil.copy(os.path.join(image_directory, image_name), os.path.join(train_img_dir, image_name))

    # Split masks into train/val sets
    # total_masks = len(masks)
    # train_masks = random.sample(masks, int(train_ratio * total_masks))
    # val_masks = [mask for mask in masks if mask not in train_masks]

    # # Process each mask and generate YOLO annotations
    # def process_mask(mask, label_dir, img_dir):
    #     # Mask's .xyn should contain [class, x_center, y_center, width, height] (normalized)
    #     yolo_annotation = []
    #     for annotation in mask.xyn:
    #         class_id, x, y, w, h = annotation
    #         yolo_annotation.append(f"{class_id} {x} {y} {w} {h}\n")
        
    #     # Save the annotation to a text file
    #     img_name = image_name.replace('.jpg', '.txt')  # Assuming image file name corresponds
    #     with open(os.path.join(label_dir, img_name), 'w') as f:
    #         f.writelines(yolo_annotation)
        
    #     # Copy the image to the correct folder (train or val)
    #     shutil.copy(os.path.join(image_directory, image_name), os.path.join(img_dir, image_name))

    # # Process training masks
    # for mask in train_masks:
    #     process_mask(mask, train_label_dir, train_img_dir)

    # # Process validation masks
    # for mask in val_masks:
    #     process_mask(mask, val_label_dir, val_img_dir)
for image_name in os.listdir(image_directory):
    if image_name.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        image_path = os.path.join(image_directory, image_name)
        image_src, image_transformed, img_size = grounding_dino_eye.DataLoader(image_path)
        boxes, logits, phrases = grounding_dino_eye.predict(
            model=groundingdino_model, 
            image=image_transformed, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE
        )
        absolute_boxes = grounding_dino_eye.convert_boxes(boxes, (img_size[1],img_size[0]))
        results = model(image_src, bboxes=absolute_boxes)
        convert_mask_to_yolo_format(results[0].masks, image_name, 'datasets')



        