import streamlit as st
from streamlit_chat import message
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import safetensors
import torch

st.title("Chatbot")

@st.cache_resource(show_spinner=True)
def convert_safetensors_to_bin(peft_model_id):
    safetensors_path = os.path.join(peft_model_id, "adapter_model.safetensors")
    bin_path = os.path.join(peft_model_id, "adapter_model.bin")
    
    if not os.path.exists(bin_path) and os.path.exists(safetensors_path):
        pt_state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
        torch.save(pt_state_dict, bin_path)
        st.success("Converted safetensors to bin.")

@st.cache_resource(show_spinner=True)
def load_model_tokenizer():
    peft_model_id = "flan-t5-large-multipurpose"
    convert_safetensors_to_bin(peft_model_id)
    
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id).to("cpu")
    
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=256).input_ids.to("cpu")
    outputs = model.generate(input_ids=input_ids, top_p=0.9, max_length=256)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

if "history" not in st.session_state:
    st.session_state.history = []

message("Hi I am Flan T5 Chatbot. How can I help you?", is_user=False)

placeholder = st.empty()
input_ = st.text_input("Human")

if st.button("Generate"):
    if input_:
        st.session_state.history.append({"message": input_, "is_user": True})
        input_ = "Human: " + input_ + ". Assistant: "
        with st.spinner(text="Generating Response.....  "):
            response = inference(model, tokenizer, input_)
            st.session_state.history.append({"message": response, "is_user": False})

for chat in st.session_state.history:
    message(chat["message"], is_user=chat["is_user"])
