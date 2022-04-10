import streamlit as st
import random
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.types import Integer
from streamlit.report_thread import get_report_ctx
#from streamlit.report_thread import add_script_run_ctx
import pydeck as pdk
from datasets import load_dataset
import pymongo
from pymongo import MongoClient
import codecs
import pickle
import json
import requests
import math
import os
import re
import certifi

# Load NTDB-GPT-2 model through huggingface API
API_URL = "https://api-inference.huggingface.co/models/dracoglacius/NTDB-GPT2"
headers = {"Authorization": "Bearer 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"}

eval_file = open("clinical_eval.json", 'r')
data = json.loads(eval_file.read())


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

# Get validation sequences path
#valid_seqs_name = "valid_noestart.txt"
#valid_seqs_path = os.path.join(os.getcwd(), valid_seqs_name)

# Load ICD9 dictionaries for sequence translations
dcode_dict_name = "dcode_dict.txt"
pcode_dict_name = "pcode_dict.txt"
dcode_dict_path = os.path.join(os.getcwd(), dcode_dict_name)
pcode_dict_path = os.path.join(os.getcwd(), pcode_dict_name)

with open(dcode_dict_path, "rb") as fp:   
    icd9_dcode_dict = pickle.load(fp)

with open(pcode_dict_path, "rb") as fp: 
    icd9_pcode_dict = pickle.load(fp)

# Load validation dataset to sample/generate from
#start_idx = 0
#end_idx = 10
#datasets = load_dataset("text", data_files={"validation": valid_seqs_path})

#datasetseq = list(datasets['validation'][:].values())[0]
#datasetseq = datasetseq[:1000]

#dataset_len = len(datasetseq)
if 'data' not in st.session_state:
    st.session_state.data = [val for val in data]
    random.shuffle(st.session_state.data)
    st.session_state.labels = [val['label'] for val in st.session_state.data] # will be sent to database
    st.session_state.seqs = [val['seq'] for val in st.session_state.data]
    #st.session_state.rated_seqs = []
    

#cleaned_datasetseq = [seq.split() for seq in datasetseq]

# MongoDB connection for sending userfeedback to 
client = pymongo.MongoClient("mongodb+srv://stemmler:project@stemmlerproject.zqsgu.mongodb.net/StemmlerProject?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true", tlsCAFile=certifi.where())
#db = client.test

# Get the database URL for heroku postgres
DATABASE_URL = os.environ['DATABASE_URL'] # comment if testing locally

# Get a unique session ID that can used at postgres primary key 
def get_session_id() -> str:
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id # postgres name convention
    return session_id

# Functions to read/write states of user input and dataframes
def write_state(column, value, engine, session_id):
    engine.execute("UPDATE %s SET %s='%s'" % (session_id, column, value))

def write_state_df(df:pd.DataFrame, engine, session_id):
    df.to_sql('%s' % (session_id),engine,index=False,if_exists='replace',chunksize=1000, dtype={str(session_id): Integer()})

def read_state(column, engine, session_id):
    state_var = engine.execute("SELECT %s FROM %s" % (column, session_id))
    state_var = state_var.first()[0]
    return state_var

def read_state_df(engine, session_id):
    try:
        df = pd.read_sql_table(session_id, con=engine)
    except:
        df = pd.DataFrame([])
    return df

# Retrieve session ID
session_id = get_session_id()
if 'mongo_data' not in st.session_state:
    st.session_state.mongo_data = {session_id:[]}

# Translation functions
def translate_dpcode_seq(list_seq, pstart_idx, dcode_dict, pcode_dict):
    translation = []
    for dcode in list_seq[:pstart_idx]:
        dcode = re.sub('[.]', '', dcode)
        if dcode in dcode_dict:
            translation.append(dcode_dict[dcode])
    for pcode in list_seq[pstart_idx+1:]:
        pcode = re.sub('[.]', '', pcode)
        if pcode in pcode_dict:
            translation.append(pcode_dict[pcode])
    return translation

def remove_period_from_seq_and_translate(list_seq, translations, dcode_dict, pcode_dict): 
    if '<PSTART>' in list_seq:
        pstart_idx = list_seq.index('<PSTART>')
    else:
        pstart_idx = len(list_seq) - 1
    translation = translate_dpcode_seq(list_seq, pstart_idx, dcode_dict, pcode_dict)
    translations.append(translation)

#for seq in cleaned_datasetseq:
#    my_translations.append(seq[1:-1])
#    remove_period_from_seq_and_translate(seq, my_translations, icd9_dcode_dict, icd9_pcode_dict)

if __name__ == '__main__':
 
    feedback_db = "user_feedback"
    engine = create_engine(DATABASE_URL, connect_args={'sslmode':'require'}) # uncomment along with line 75 for deployment
    #engine = create_engine('sqlite:///testDB.db') # comment when done with local changes
    mongo_db = client[feedback_db] # user_feedback DB that all feedback is sent to
    mongo_feedback_collection = mongo_db[session_id] # each person's session ID is used to create a collection inside the feedback DB

    # create state tables of session
    engine.execute("CREATE TABLE IF NOT EXISTS %s (size text)" % (session_id))
    len_table = engine.execute("SELECT COUNT(*) FROM %s" % (session_id))
    len_table = len_table.first()[0]
    if len_table == 0:
        engine.execute("INSERT INTO %s (size) VALUES ('1')" % (session_id))

    # can now create pages
    page = st.sidebar.selectbox("Select page:", ("About", "NTDB-GPT-2","Inference","Evaluation"))

    # Import README markdown file 
    read_me_file_name = "README.md"
    read_me_file_path = os.path.join(os.getcwd(), read_me_file_name)
    read_me_file = open(read_me_file_path)
    read_me = read_me_file.read()
    
    if page == "About":
        st.markdown(read_me)

    elif page == "NTDB-GPT-2":
        st.header("Work in progress")

    elif page == "Inference":
        st.subheader("Here are some sample sequences to generate from:")
        #for seq in cleaned_datasetseq[3:]:
            #st.text(' '.join(seq))
        query_str = st.text_input("Enter a sequence stub starting with <START> ECODE ...")
        if st.button("Generate"):
            output = query(query_str)
            output_tt = output[0].values()
            output_tt = list(output_tt)
            output_tt = output_tt[0].split(' ')
            translations = []
            #translations.append(output_tt[1:-1])
            remove_period_from_seq_and_translate(output_tt, translations, icd9_dcode_dict, icd9_pcode_dict)
            st.write(output[0])
            st.write(translations[-1])
        
    elif page == "Evaluation":
        clin_loe = st.selectbox('What is your clinical level of education?',
        ('Medical Student', 'Resident/Fellow', 'Attending'))
        st.subheader("Given the following stem consisting of an presenting injury (START) and diagnosis (DXS) do the following procedures (PRS) make clinical sense? Rate the 3 sequences below! (If no procedures are listed, is lack of surgical intervention a valid outcome?)\n")
        col_1, col_2 = st.columns(2)
        #seq_samples = random.sample(range(dataset_len), 3)
        #for idx, seq in enumerate(st.session_state.seq_samples):
        #   st.session_state.seq_samples[idx] += 1
        #st.write(seq_samples)
        #idxs = []
        rated_seqs = []
        for idx, seq in enumerate(st.session_state.seqs[:5],1):
            #idxs.append(idx)
            #list_seq = my_translations[idx] 
            #list_trans = my_translations[idx+1]
            rated_seqs.append(str(st.session_state.labels[idx-1])+"_seq_:"+str(seq)+"_realistic_:")
            #rated_trans.append(list_trans)
            #input_seq = list_seq[:3]
            #input_seq = ' '.join(input_seq)
            with st.container():
                with col_1:
                    st.header(f"Sequence {idx}:")
                    st.text(seq) # exclude start/end in output
                    #print(str(list_seq))
                #with col_2:
                #    st.text(st.session_state.seqs[:5])
                #    seq_1_plaus = st.checkbox("Realistic?")
                #    seq_2_plaus = st.checkbox("Realistic?")
                #    seq_3_plaus = st.checkbox("Realistic?")

                    #st.header(f"Translation {idx}:")
                    #st.write(list_trans)
        form = st.form("checkboxes", clear_on_submit = True)
        with form:
            seq_1_plaus = st.checkbox(f"Is sequence 1 realistic?")
            seq_2_plaus = st.checkbox(f"Is sequence 2 realistic?")
            seq_3_plaus = st.checkbox(f"Is sequence 3 realistic?")
            seq_4_plaus = st.checkbox(f"Is sequence 4 realistic?")
            seq_5_plaus = st.checkbox(f"Is sequence 5 realistic?")
        submit = form.form_submit_button("Submit your answers")

        #write_state("size", data, engine, session_id)
        #size = int(read_state("size", engine, session_id))

        if submit:
            rated_seqs[0] += str(seq_1_plaus)
            rated_seqs[1] += str(seq_2_plaus)
            rated_seqs[2] += str(seq_3_plaus)
            rated_seqs[3] += str(seq_4_plaus)
            rated_seqs[4] += str(seq_5_plaus)
            st.session_state.mongo_data[session_id].extend(rated_seqs)
            data = {'clin_LOE':clin_loe, 'rated_seqs':st.session_state.mongo_data[session_id]}
            df = pd.DataFrame(data)
            write_state_df(df, engine, session_id + "_df") 
            

            # Update with upsert to create new document if one doesn't exist
            #{"all_feedback":st.session_state.mongodata}
            mongo_feedback_collection.update_one(
                {}, # empty doc returns first doc in collection
                {"$set": data},
                upsert=True
            )

        #if not (read_state_df(engine, session_id + "_df").empty):
            #df = read_state_df(engine, session_id + "_df")
            del st.session_state.seqs[:5]
            del st.session_state.labels[:5]
            st.experimental_rerun()
            st.success("Feedback successfully submitted!")
            st.write(df.astype(str))
