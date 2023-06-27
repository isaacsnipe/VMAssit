import os.path

from pydantic import BaseSettings

from glob import glob
import re
import logging
import time 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import shutil

from typing import Any


from pydantic import BaseSettings

from core.pydoctr import OcrPDF, ocrpdf
from core.ner_flair import FlairModel, flair_model

import ocrmypdf

from core.ocr import run_ocrmypdf

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import os

import torch

from PyPDF2 import PdfFileReader

from fuzzywuzzy.process import dedupe
from fuzzywuzzy import fuzz


class VLAModel:
    def __init__(
        self, model_classification_checkpoint, model_extraction_checkpoint
    ) -> None:
        self.PDF_FILE_PATH = "core/temp_pdf.pdf"
        self.OCR_PDF_FILE_PATH = "core/temp_ocr_pdf.pdf"
        self.model_classification_checkpoint = model_classification_checkpoint
        self.model_extraction_checkpoint = model_extraction_checkpoint
        self.flair_model = None

        
    def initialize_model(self):
        if self.flair_model is None:
        # self.ocrpdf = OcrPDF()
        # self.flair_model = FlairModel('flair/ner-english-ontonotes-large')
        
            self.ocrpdf = ocrpdf
            self.flair_model = flair_model
            
            self.model_classification_tokenizer = AutoTokenizer.from_pretrained(
                self.model_classification_checkpoint
            )
            self.model_classification = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_classification_checkpoint
            )  # .to("cuda")

            
            self.model_extraction_tokenizer = AutoTokenizer.from_pretrained(
                self.model_extraction_checkpoint
            )
            self.model_extraction = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_extraction_checkpoint
            )  # .to("cuda")
        


    def _save_pdf(self, pdf) -> None:
        destination = self.PDF_FILE_PATH
        try:
            with open(destination, "wb+") as file_object:
                file_object.write(pdf.file.read())
        finally:
            pdf.file.close()

    def _remove_pdfs(self) -> None:
        """
        Search for all .pdf files and delete them
        """
        for f in glob(".\\core\\*.pdf"):
            print(f)
            if os.path.exists(f):
                os.remove(f)

    def _load_list_of_texts_from_pages(self, pdf_file_path):

        start_time = time.time()
        # list_of_texts_from_pages = self.ocrpdf.perform_ocr(pdf_file_path)
        # run_ocrmypdf(pdf_file_path, self.OCR_PDF_FILE_PATH)
        ocrmypdf.ocr(pdf_file_path, self.OCR_PDF_FILE_PATH, plugins=['core/ocrmypdf_plugin.py'], force_ocr=True, grayscale=True)
                
        # return list_of_texts_from_pages
        pdf = PdfFileReader(self.OCR_PDF_FILE_PATH)

        list_of_texts_from_pages = []
        for i in range(pdf.numPages):
            list_of_texts_from_pages.append(pdf.pages[i].extract_text())
        logger.info(f"OCR Processing time: {time.time() - start_time}")

        return list_of_texts_from_pages

    def _inference_sample(self, model, tokenizer, text, max_length=16000):
        input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )  # .to("cuda")  # Batch size 1
        model.eval()

        generated_ids = model.generate(
            input_ids=input_ids,
            num_beams=1,
            do_sample=False,
            max_length=80,
            repetition_penalty=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        probs = torch.stack(generated_ids.scores, dim=1).softmax(-1)
        t = [x.topk(1) for x in probs][0].values.squeeze(-1)
        confidence = torch.prod(t, 0).item()

        tokens = generated_ids.sequences.squeeze()
        prediction = tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return prediction, round(confidence, 4)

    def _predict_classification(self):
        prediction, confidence = self._inference_sample(
            self.model_classification,
            self.model_classification_tokenizer,
            "classification : " + self.whole_document,
            max_length=16000,
        )

        return {"prediction": prediction, "confidence": round(confidence, 4)}
    
    # Edit to use whole document instead of list of pages
    def _predict_extraction_for_list_of_pages(self, task, probability_tresh=0):
        list_of_preds = []

        prediction, confidence = self._inference_sample(
            self.model_extraction,
            self.model_extraction_tokenizer,
            task + " : " + self.whole_document,
            # max_length=512,
        )
        list_of_preds.append(
            {
                "prediction": prediction,
                "confidence": round(confidence, 4),
            }
        )
        list_of_preds = [x for x in list_of_preds if x["prediction"] != "None"]
        list_of_preds = [
            x for x in list_of_preds if x["confidence"] > probability_tresh
        ]

        return list_of_preds

    # def _predict_extraction_for_list_of_pages(self, task, probability_tresh=0):
    #     list_of_preds = []
    #     for i in range(len(self.list_of_pages)):
    #         prediction, confidence = self._inference_sample(
    #             self.model_extraction,
    #             self.model_extraction_tokenizer,
    #             task + " : " + self.list_of_pages[i],
    #             max_length=512,
    #         )
    #         list_of_preds.append(
    #             {
    #                 "prediction": prediction,
    #                 "confidence": round(confidence, 4),
    #                 "page_number": i + 1,
    #             }
    #         )
    #     list_of_preds = [x for x in list_of_preds if x["prediction"] != "None"]
    #     list_of_preds = [
    #         x for x in list_of_preds if x["confidence"] > probability_tresh
    #     ]

    #     return list_of_preds

    def predict(self, document, filename):
        self._remove_pdfs()

        self._save_pdf(document)
        #self.whole_document = self.ocrpdf.perform_ocr(self.PDF_FILE_PATH)
    
        self.list_of_pages = self._load_list_of_texts_from_pages(self.PDF_FILE_PATH)
        self.whole_document = " ".join(self.list_of_pages)
        self.predictions = {}
        
        # Adding filename from uploadfile parameter
        self.predictions["file"] = {
            "name" : filename
        }

        start_time = time.time()
        if self._predict_classification():
            self.predictions["label"] = self._predict_classification()
        else:
            self.predictions["label"] = ["None"]
        logger.info(f"Prediction time of document label: {time.time() - start_time}")

        # Use Flair model and postprocessing to extract the end date
        end_date = self.flair_model.predict_entities("contract_end_date", self.whole_document)
        
        if end_date:
            self.predictions["contract_end_date"] = {
                "prediction": end_date
            }
        else:
            self.predictions["contract_end_date"] = {
                "prediction": [
                    {"prediction": "None", "confidence": "None", "page_number": "None"}
                ]
            }
        logger.info(f"Prediction time of document end date: {time.time() - start_time}")
        
        start_time = time.time()
        # Use Flair model and postprocessing to extract the end date
        start_date = self.flair_model.predict_entities("contract_start_date", self.whole_document, self.OCR_PDF_FILE_PATH)
        
        if start_date:
            self.predictions["contract_start_date"] = {
                "prediction": start_date
            }
        else:
            self.predictions["contract_start_date"] = {
                "prediction": [
                    {"prediction": "None", "confidence": "None", "page_number": "None"}
                ]
            }
        logger.info(f"Prediction time of contract start date: {time.time() - start_time}")

        start_time = time.time()
        # Use regex patterns to extract all site-ids
        site_id_pattern = r'\b[A-Za-z]{2}[_\s-]\d{4}[a-zA-Z]?\b'
        
        site_id_preds_list = re.findall(site_id_pattern, self.whole_document)
        print(site_id_preds_list)
        
        if site_id_preds_list:
            site_id_preds = site_id_preds_list[0] 
        else:
            site_id_preds = site_id_preds_list

        self.predictions["site_id"] = {"prediction": site_id_preds}
        logger.info(f"Prediction time of document site-id: {time.time() - start_time}")

        start_time = time.time()
        # Keep with Jovan's model
        # if self._predict_extraction_for_list_of_pages(
        #     "parties", probability_tresh=0.37
        # ):
        #     self.predictions["parties"] = self._predict_extraction_for_list_of_pages(
        #         "parties", probability_tresh=0.37
        #     )
        #     ppreds = sorted(
        #         self.predictions["parties"], key=lambda d: d["confidence"], reverse=True
        #     )

        #     parties_final_list = []
        #     for i in range(len(ppreds)):
        #         if ";" in ppreds[i]["prediction"]:
        #             parties_final_list.extend(ppreds[i]["prediction"].split(";"))
        #         else:
        #             parties_final_list.append(ppreds[i]["prediction"])
        #     parties_final_list = [x.strip() for x in parties_final_list]

        #     self.predictions["parties"] = {
        #         "prediction": ";".join(
        #             list(dedupe(parties_final_list, threshold=80, scorer=fuzz.ratio))
        #         )
        #     }
        involved_parties = self.flair_model.predict_entities("parties", self.whole_document, self.OCR_PDF_FILE_PATH)
        if involved_parties:
            self.predictions["parties"] = {
                "prediction": involved_parties
            }
        else:
            self.predictions["parties"] = {"prediction": "None"}
        logger.info(f"Prediction time of involved parties: {time.time() - start_time}")

        return self.predictions

    def predict_file_path(self, pdf_file_path):
        self.list_of_pages = self._load_list_of_texts_from_pages(pdf_file_path)
        self.whole_document = " ".join(self.list_of_pages)
        self.predictions = {}

        self.predictions["class"] = self._predict_classification()

        self.predictions[
            "contract_end_date"
        ] = self._predict_extraction_for_list_of_pages(
            "contract_end_date", probability_tresh=0
        )

        site_id_preds = self._predict_extraction_for_list_of_pages(
            "site_id", probability_tresh=0
        )
        site_id_preds = [x["prediction"] for x in site_id_preds]
        site_id_preds = list(set(site_id_preds))

        self.predictions["site_id"] = {"prediction": site_id_preds}

        self.predictions["parties"] = self._predict_extraction_for_list_of_pages(
            "parties", probability_tresh=0.37
        )

        ppreds = sorted(
            self.predictions["parties"], key=lambda d: d["confidence"], reverse=True
        )

        parties_final_list = []
        for i in range(len(ppreds)):
            if ";" in ppreds[i]["prediction"]:
                parties_final_list.extend(ppreds[i]["prediction"].split(";"))
            else:
                parties_final_list.append(ppreds[i]["prediction"])
        parties_final_list = [x.strip() for x in parties_final_list]

        self.predictions["parties"] = {
            "prediction": list(
                dedupe(parties_final_list, threshold=80, scorer=fuzz.ratio)
            )
        }

        return self.predictions


class Settings(BaseSettings):
    APP_NAME: str = "VirtualLegalAssistant"
    version: str = "1.0"
    releaseId: str = "1.1"
    API_V1_STR: str = "/api/v1"
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


# NOTE: uncomment this when you are running it on local or on prem
MODEL_CLASSIFICATION_CHECKPOINT = '/home/ghana/VirtualLegalAssistant/VLA_API/core/checkpoint-768'
MODEL_EXTRACTION_CHECKPOINT = '/home/ghana/VirtualLegalAssistant/VLA_API/core/checkpoint-4608'

# # use this when you are running it on docker
# #MODEL_CLASSIFICATION_CHECKPOINT = "/code/core/checkpoint-768"
# #MODEL_EXTRACTION_CHECKPOINT = "/code/core/checkpoint-4608"

settings = Settings()


vla_model = VLAModel(MODEL_CLASSIFICATION_CHECKPOINT, MODEL_EXTRACTION_CHECKPOINT)
