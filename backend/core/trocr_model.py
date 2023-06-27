from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from doctr.io import read_pdf
from doctr.models import detection_predictor

# detection_model = detection_predictor(arch='db_resnet50', pretrained=True)

class TrocrModel:
    def __init__(self):
        self.detection_model = None
    
    def initialize_model(self):
        if self.detection_model is None:
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
            self.detection_model = detection_predictor(arch='db_resnet50', pretrained=True)
    
    def detect(self, pdf_path):
        array = read_pdf(pdf_path)
        return self.detection_model(array), array
    
    def predict(self, image_section):
        pixel_values = self.processor(images=image_section, return_tensors="pt").pixel_values
        generated_ids = self.trocr_model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    
    
trocrModel = TrocrModel()