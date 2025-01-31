# from autodistill_grounded_sam import GroundedSAM
# from autodistill.detection import CaptionOntology
# from autodistill_yolov8 import YOLOv8
# import supervision as sv
# import roboflow
# import random
# import cv2
# import numpy as np
# import os

# base_model = GroundedSAM(ontology=CaptionOntology({'round tabs':'round tabs'}))
# image_name = str(random.choice(sv.list_files_with_extensions(directory=r'D:\countbeat images\countbeat image set\jbcl')))
# mask_annotator = sv.MaskAnnotator()
# image = cv2.imread(image_name)
# classes = base_model.ontology.classes()
# detections = base_model.predict(image_name)
# print(detections)
# # image = cv2.imread(image_name)

# # # Ensure the image is loaded
# # if image is None:
# #     raise FileNotFoundError(f"Image not found at {image_name}")

# # # Create a copy of the image for visualization
# # output_image = image.copy()

# # # Colors for different detections
# # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red

# # # Draw detections
# # for i, (bbox, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
# #     # Extract bounding box coordinates
# #     x1, y1, x2, y2 = map(int, bbox)

# #     # Resize the corresponding mask to match the bounding box dimensions
# #     object_mask = detections.mask[i]
# #     resized_mask = cv2.resize(object_mask.astype(np.uint8), (x2 - x1, y2 - y1))

# #     # Overlay the mask on the image
# #     color = colors[i % len(colors)]
# #     for c in range(3):  # Apply the color to each channel
# #         output_image[y1:y2, x1:x2, c] = np.where(
# #             resized_mask > 0,
# #             (0.5 * output_image[y1:y2, x1:x2, c] + 0.5 * color[c]).astype(np.uint8),
# #             output_image[y1:y2, x1:x2, c]
# #         )

# #     # Draw bounding box
# #     cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

# #     # Add confidence score as text
# #     cv2.putText(output_image, f"Conf: {conf:.2f}", (x1, y1 - 10),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# # # Display the output
# # cv2.imshow("Detections with Masks", output_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # # Save the output image (optional)
# # cv2.imwrite("output_detections_with_masks.jpg", output_image)


from groundingdinoeye import Inference

image_path = r'D:\countbeat images\countbeat image set\jbcl'



grounding_dino_eye = Inference()

image_src, image= grounding_dino_eye.DataLoader(image_path)
print(image_src, image)

groundingdino_model = grounding_dino_eye.LoadModel()