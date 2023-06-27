from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# from doctr.models import kie_predictor

# model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


class OcrPDF:
    def __init__(self):
        self.model = None

    def initialize_model(self):
        if self.model is None:
            self.model = ocr_predictor(
                det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
            )
    
    def perform_ocr(self, pdf_path, show=False):
        pdf_doc = DocumentFile.from_pdf(pdf_path)
        result = self.model(pdf_doc)

        if show:
            result.show(pdf_doc)

        json_output = result.export()

        
        pages = []

        for i in range(len(json_output["pages"])):
            words = []
            for j in range(len(json_output["pages"][i]["blocks"])):
                for k in range(len(json_output["pages"][i]["blocks"][j]["lines"])):
                    for l in range(
                        len(json_output["pages"][i]["blocks"][j]["lines"][k]["words"])
                    ):
                        words.append(
                            json_output["pages"][i]["blocks"][j]["lines"][k]["words"][
                                l
                            ]["value"]
                        )
            pages.append(words)
        return pages

ocrpdf = OcrPDF()