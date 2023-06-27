from flair.data import Sentence
from flair.models import SequenceTagger
import re
import dateparser
import random
from PIL import Image
import shutil
import jellyfish
from tqdm import tqdm

from core.trocr_model import trocrModel
from fuzzywuzzy.process import dedupe

def generate_substrings(string):
        # Split the string into words
        words = string.split()

        # Generate all possible consecutive sub-strings
        substrings = []
        for i in range(len(words)):
            for j in range(i+1, len(words)+1):
                substrings.append(" ".join(words[i:j]))

        return substrings
    
def strip_dates(dates):
    new_dates = []
    for i in range(len(dates)):
        date = dates[i]
        combined = re.search(r'\d+[^0-9 /\.-]+', date, re.IGNORECASE)
        
        if combined:
            print(f"{date} was combined")
            word = re.sub(r"\d", "", combined.group(0))
            print(word)
            date = date.replace(word, "")

        new_dates.append(date)

    return new_dates

def filter_involved_parties(company_list):
    # Initialize the count dictionary
    count_dict = {}

    for company in company_list:
        company = company.lower()
        count = 0
        for other_company in company_list:
            other_company = other_company.lower()
            # Compute text similarity between two company names
            similarity = jellyfish.jaro_winkler(company, other_company)
            # If the texts are similar or one is a subset of the other, count the number of occurences
            if ((company in other_company) or (other_company in company) or (similarity>= 0.75)):
                count += 1
        count_dict[company] = count
        
    deduplicated_companies = []
    for i in range(len(count_dict.keys())):
        for j in range(i+1, len(count_dict.keys())):
            company = list(count_dict.keys())[i]
            other_company = list(count_dict.keys())[j]
            similarity = jellyfish.jaro_winkler(company, other_company)
            if ((company in other_company) or (other_company in company) or (similarity>= 0.75)):
                if other_company not in deduplicated_companies:
                    deduplicated_companies.append(other_company)
    for key in deduplicated_companies:
      del count_dict[key]     

    count_dict = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
    if len(count_dict) >= 2:
        involved_parties = " ; ".join([count_dict[0][0].upper(), count_dict[1][0].upper()])
    elif len(count_dict) == 1:
        involved_parties = " ; ".join([count_dict[0][0].upper()])
    else:
        involved_parties = None
    
    return involved_parties

def strip_date(date):
    combined = re.search(r'\d+[^0-9 /\.-]+', date, re.IGNORECASE)
    
    if combined:
        print(f"{date} was combined")
        word = re.sub(r"\d", "", combined.group(0))
        print(word)
        date = date.replace(word, "")

    return date

class FlairModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.predictions = None
        self.model = None
        
    def initialize_model(self):
        if self.model is None:
            self.model = SequenceTagger.load(self.model_name)
            self.trocr_model = trocrModel


    def predict(self, string, tags=["DATE", "ORG"], conf_thresh=0.7):
        self.tags = tags
        self.string = string
        predictions = {tag: [] for tag in self.tags}

        sentence = Sentence(string)
        self.model.predict(sentence, return_probabilities_for_all_classes=True)

        # iterate over entities and print
        for entity in sentence.get_spans("ner"):
            for tag in self.tags:
                if entity.tag == tag:
                    if entity.score > conf_thresh:
                        predictions[tag].append(entity.text)

        pattern = r"\d{2}\.\d{2}\.\d{4}"
        regex_dates = re.findall(pattern, string)
        predictions["DATE"] += regex_dates

        return predictions


    def predict_entities(self, task, string, pdf_path=""):
        self.predictions = self.predict(string)
        
        if task == "parties":
            self.parties = self.predictions.get("ORG")
            return filter_involved_parties(self.parties)
        
        elif task == "contract_end_date":
            self.dates = self.predictions.get("DATE")
            self.dates = strip_dates(self.dates)
            reference_date = dateparser.parse("12/12/1800")
            date_settings = {"RELATIVE_BASE": reference_date}
            self.dates = [dateparser.parse(sub_string, settings=date_settings)  
                          for date in self.dates 
                          for sub_string in generate_substrings(date) 
                          if dateparser.parse(sub_string)]
            date_list = [date for date in self.dates if date < dateparser.parse('31st December 2050')]
            if len(date_list) > 0:
                end_date = max(date_list)
            else:
                end_date = None
            
            # self.dates = [re.sub(r"(?<=\d)[A-Za-z]+", "", date) for date in self.predictions.get("DATE")]
            # reference_date = dateparser.parse("12/03/1960")

            # self.dates = [
            #     dateparser.parse(date, settings={"RELATIVE_BASE": reference_date})
            #     for date in self.dates
            # ]
            # self.dates = [date for date in self.dates if date is not None]
            

            # greatest_date = None

            # for date_string in self.dates:
            #     if date_string is None:
            #         continue
            #     if (
            #         (greatest_date is None)
            #         or (date_string > greatest_date)
            #         and ((date_string.year) < 2100)
            #     ):
            #         greatest_date = date_string

            return end_date
        
        elif task == "contract_start_date":
            # regex patterns
            # Date and word then a date (eg: 12/03/2023 to 12/20/2023)
            pattern1 = r"(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})\s*\w+\s*(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})"
            pattern2 = r"(\d{2}[^a-zA-Z0-9/]*\s+\w+\s+\d{2,})\s*\w+\s*(\d{2}[^a-zA-Z0-9/]*\s+\w+\s+\d{2,})"
            pattern3 = r"(\d{2}[^a-zA-Z0-9/]*\s+\w+\s+\d{2,})\s*\w+\s*(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})"
            pattern4 = r"(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})\s*\w+\s*(\d{2}[^a-zA-Z0-9]*\s+\w+\s+\d{2,})"

            # compile regex patterns
            regex1 = re.compile(pattern1)
            regex2 = re.compile(pattern2)
            regex3 = re.compile(pattern3)
            regex4 = re.compile(pattern4)

            # search for patterns in document
            matches1 = regex1.findall(string)
            matches2 = regex2.findall(string)
            matches3 = regex3.findall(string)
            matches4 = regex4.findall(string)

            matches = matches1 + matches2 + matches3 + matches4

            if matches:
                date_list = [dateparser.parse(date) for date in matches[0]]
                start_date = date_list[0]
                return start_date
            else:              
                pattern_double_1 = r"(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})\s*/\s*(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})"
                pattern_double_2 = r"(\d{2}[^a-zA-Z0-9/]*\s+\w+\s+\d{2,})\s*/\s*(\d{2}[^a-zA-Z0-9/]*\s+\w+\s+\d{2,})"
                pattern_double_3 = r"(\d{2}[^a-zA-Z0-9/]*\s+\w+\s+\d{2,})\s*/\s*(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})"
                pattern_double_4 = r"(\d{2}[^a-zA-Z0-9]\d{2}[^a-zA-Z0-9]\d{2,})\s*/\s*(\d{2}[^a-zA-Z0-9]*\s+\w+\s+\d{2,})"

                # compile regex patterns
                regex1 = re.compile(pattern_double_1)
                regex2 = re.compile(pattern_double_2)
                regex3 = re.compile(pattern_double_3)
                regex4 = re.compile(pattern_double_4)

                # search for patterns in document
                matches1 = regex1.findall(string)
                matches2 = regex2.findall(string)
                matches3 = regex3.findall(string)
                matches4 = regex4.findall(string)

                matches = matches1 + matches2 + matches3 + matches4

                if matches:
                    date_list = [str(dateparser.parse(date)) for date in random.sample(matches, 1)[0]]
                    start_date = " and ".join(date_list)
                    return start_date
                
                else:
                    start_date = self.dates[0]
                
                # else:
                #     text_list = []    
                #     bounding_boxes, array = self.trocr_model.detect(pdf_path)
                    
                #     for i in tqdm(range(len(bounding_boxes))):
                #         for j in range(len(bounding_boxes[i])):
                #             left, upper, right, lower, confidence = tuple(bounding_boxes[i][j][:]) 
                #             img = Image.fromarray(array[i])
                #             width, length = img.size
                #             box = left*width, upper*length, right*width, lower*length
                #             section = img.crop(box)
                #             l, w = section.size
                #             if w > 20 and confidence < 0.75:
                #                 generated_text = self.trocr_model.predict(section)
                #                 text_list.append(generated_text)
                                
                #     pattern = r"\d+"
                #     text_list = [s for s in text_list if re.search(pattern, s)]
                    
                #     conf_thresh = 0.7
                #     date_list = []
                #     for string in text_list:
                #         sentence = Sentence(string)
                #         self.model.predict(sentence, return_probabilities_for_all_classes=True)

                #         # iterate over entities and print
                #         for entity in sentence.get_spans("ner"):
                #             for tag in ["DATE"]:
                #                 if entity.tag == tag:
                #                     if entity.score > conf_thresh:
                #                         date_list.append(entity.text)

                #     date_list = [dateparser.parse(date) for date in date_list]
                #     if date_list:
                #         # return "and".join(date_list)
                #         return date_list[0]
                #     else:
                #         return "No Start Date"
        
        else:
            return ""


flair_model = FlairModel('flair/ner-english-ontonotes-large')