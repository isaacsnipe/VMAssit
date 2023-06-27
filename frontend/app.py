import streamlit as st 
import requests 
import xlsxwriter 
from io import BytesIO 
from tqdm import tqdm
import pandas as pd

output = BytesIO() 
#st.image("4th-IR_Horizontal.png") 
st.header("4th-IR Esqr (VLA)")
st.write("Expert System for Query Retrieval") 
input_file = st.file_uploader("Upload Files", accept_multiple_files=True) 
csv_file_name = st.text_input("What do you want your file to be called?") 
input_file_count = len(input_file) 
df = pd.DataFrame(columns=["Site_ID", "Label", "Contract Start Date", "Contract End Date", "Involved Parties", "File Name"])
row = 1 

if input_file is not None and csv_file_name is not None and st.button("Evaluate"): 
    progress_bar = st.progress(0)
    for i in range(input_file_count): 
        interim = (input_file[i]) 
        files = [('document', (interim.name, interim.getvalue(), 'application/pdf'))] 
        response = requests.post('http://192.168.40.11:8183/api/v1/classify', files=files)
        site_id = response.json()["site_id"]["prediction"]
        label = response.json()["label"]["prediction"]
        contract_start_date = response.json()["contract_start_date"]["prediction"]
        contract_end_date = response.json()["contract_end_date"]["prediction"]
        party = response.json()["parties"]["prediction"]
        file_name = response.json()["file"]["name"]
        df.loc[i, "Site_ID"] = site_id
        df.loc[i, "Label"] = label
        df.loc[i, "Contract Start Date"] = contract_start_date
        df.loc[i, "Involved Parties"] = party
        df.loc[i, "Contract End Date"] = contract_end_date
        df.loc[i, "File Name"] = file_name
        df.to_csv(f'{csv_file_name}.csv')        
        progress_bar.progress(((i + 1)/input_file_count), text=f'File {i+1} out {input_file_count}')
    # Download results
    csv_data = df.to_csv()
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results",
        data=csv_data,
        file_name=f"{csv_file_name}.csv",
        mime="text/csv"
    )
