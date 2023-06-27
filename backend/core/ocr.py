
import subprocess
from subprocess import DEVNULL, STDOUT, check_call


PLUGIN_PATH = "./ocrmypdf_plugin.py"
OUTPUT_TXT_PATH = "./temp.txt"
DEBUG_TXT_PATH = "./debug_temp.txt"


def run_ocrmypdf(input_pdf_path,output_pdf_path):


    bashCommand = f"""ocrmypdf 
                        -l eng+fra+deu+ita
                        --output-type pdf
                        
                        --plugin {PLUGIN_PATH} --grayscale-ocr
                        --rotate-pages
                        --deskew
                        
                        --force-ocr
                        --clean
                        --sidecar 
                        {OUTPUT_TXT_PATH}
                        {input_pdf_path}
                        {output_pdf_path}"""    

    try:
        
        process = subprocess.Popen(bashCommand.split(),stdout=subprocess.DEVNULL ,stderr=subprocess.PIPE)  #shell=False, stdout=DEVNULL  stdout=subprocess.PIPE
        output, error = process.communicate()
        
        with open(DEBUG_TXT_PATH, "w") as text_file:
            text_file.write(error.decode())        
    except:
        pass

# if __name__ == "__main__":

#     """ This is an usage example of how to run a library called "ocrmypdf". 
#         Input is a pdf file, and output is another pdf file with new OCR text embeded. 
#         Change the name of the output_txt_path so that it coresponds to input name.
#         Leave bash command options as is. 
#         The function also outputs "./temp.txt". You can erase this, or leave it, function will overright it on the next call. 
#         Also you have the "./debug_temp.txt", which is information usefull for debuging. It would be good to save these somewhere for each pdf separately.
#     """

#     input_pdf_path = '/home/jovis/Documents/WORK/legal_assistant/OLD/documents/uncompresed/AG/AA_Inactive_Sites/AG_7560/Contracts/2_AG_7560_A_11.12.01_Contract.pdf'
#     output_pdf_path = "./temp.pdf"

#     run_ocrmypdf(input_pdf_path,output_pdf_path)