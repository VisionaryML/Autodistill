import transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import torch
from slconfig import SLConfig
from Dino import build_groundingdino
from collections import OrderedDict
from typing import Tuple, Any, Dict, List
import bisect
from transformers import AutoTokenizer
from torchvision.ops import box_convert
import supervision as sv
import cv2


class Inference:
    
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    def __init__(self):
        print("init done.")
    
    def DataLoader(self, image_path: str):
        transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),  # This needs to handle only the image
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.open(image_path).convert("RGB")  # Open image
        image = np.asarray(image_source)  # Convert to numpy array if needed
        
        # Apply the transformations correctly
        image_transformed, _ = transform(image_source, None) 
        return image, image_transformed, image_source.size
    
    def clean_state_dict(self,state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == "module.":
                k = k[7:]  # remove `module.`
            new_state_dict[k] = v
        return new_state_dict
    
    def LoadModel(self,device='cuda'):
        cache_config_file = hf_hub_download(repo_id=self.ckpt_repo_id, filename=self.ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file) 
        model = build_groundingdino(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=self.ckpt_repo_id, filename=self.ckpt_filenmae)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(self.clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model
    
    def preprocess_caption(caption_data: str) -> str:
        result = caption_data.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
    
    def get_phrases_from_posmap(
            posmap: torch.BoolTensor, tokenized: Dict, tokenizer: AutoTokenizer, left_idx: int = 0, right_idx: int = 255
    ):
        assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
        if posmap.dim() == 1:
            posmap[0: left_idx + 1] = False
            posmap[right_idx:] = False
            non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
            token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
            return tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")

    
    def predict(
            self,
            model,
            image: torch.Tensor,
            caption: str,
            box_threshold: float,
            text_threshold: float,
            device: str = "cuda",
            remove_combined: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        caption = Inference.preprocess_caption(caption_data=caption)

        model = model.to(device)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        
        if remove_combined:
            sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
            
            phrases = []
            for logit in logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(Inference.get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
        else:
            phrases = [
                Inference.get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit
                in logits
            ]

        return boxes, logits.max(dim=1)[0], phrases
    
    def annotate(self,image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
        """    
        This function annotates an image with bounding boxes and labels.

        Parameters:
        image_source (np.ndarray): The source image to be annotated.
        boxes (torch.Tensor): A tensor containing bounding box coordinates.
        logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
        phrases (List[str]): A list of labels for each bounding box.

        Returns:
        np.ndarray: The annotated image.
        """
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)

        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]

        bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame
    
    def convert_boxes(self, boxes, image_size):
        """
        Convert normalized [x_center, y_center, w, h] boxes to absolute [x1, y1, x2, y2].
        
        Args:
            boxes (torch.Tensor): Tensor of shape (N, 4) with normalized boxes.
            image_size (tuple): Original image dimensions (height, width).
        
        Returns:
            torch.Tensor: Converted boxes in absolute coordinates (N, 4).
        """
        h, w = image_size
        boxes = boxes.clone()  # Avoid modifying original tensor
        
        # Scale normalized coordinates to image size
        boxes[:, [0, 2]] *= w  # x_center and width
        boxes[:, [1, 3]] *= h  # y_center and height
        
        # Convert center to corners
        boxes[:, 0] -= boxes[:, 2] / 2  # x_min = x_center - width/2
        boxes[:, 1] -= boxes[:, 3] / 2  # y_min = y_center - height/2
        boxes[:, 2] += boxes[:, 0]      # x_max = x_min + width
        boxes[:, 3] += boxes[:, 1]      # y_max = y_min + height
        
        return boxes.int()


    



