import gradio as gr
from gradio_log import Log
import random
import numpy as np
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from peft import PeftModel
import torch

import time
from threading import Thread

import logging
from pathlib import Path

torch.random.manual_seed(1)

class CustomFormatter(logging.Formatter):

    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


formatter = CustomFormatter()

log_file = "./log_chatbot.txt"
Path(log_file).touch()

ch = logging.FileHandler(log_file)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger = logging.getLogger("gradio_log")
logger.setLevel(logging.DEBUG)
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.addHandler(ch)


logger.info("The logs will be displayed in here.")

def create_log_handler(level):
    def l(text):
        getattr(logger, level)(text)

    return l

# model_id = "/home/stefanwebb/models/llm/meta_llama3-8b"
model_id = "/home/stefanwebb/models/llm/meta_llama3-8b-instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='cuda',
    torch_dtype="auto"
)

# peft_model_id = "/home/stefanwebb/code/python/test-qwen2/stefans-debug-llama3-chat-bs-16-eval/checkpoint-6876"

peft_model_id = "/home/stefanwebb/code/python/train-sentence-classifier/stefans-debug-llama3-sentence-classifier/checkpoint-864"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# model = base_model

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='right')
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = "<|reserved_special_token_0|>"
# tokenizer.pad_token_id = 128002
# tokenizer.eos_token = "<|eot_id|>"
# tokenizer.eos_token_id = 128009

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def llama3_format_prompt(s):
        return f"<|start_header_id|>user<|end_header_id|>\n\n{s.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def llama3_format_response(s):
        return f"{s.strip()}<|eot_id|>"

# def sample_response(message):
#     inputs = tokenizer(message, return_tensors="pt")
#     input_ids = inputs["input_ids"].cuda()
#     generation_output = model.generate(
#         input_ids=input_ids,
#         generation_config=GenerationConfig(do_sample=True, temperature=0.1, top_p=0.75, num_beams=4),
#         return_dict_in_generate=True,
#         output_scores=True,
#         max_new_tokens=512,
#         tokenizer=tokenizer,
#         stop_strings=["<end_of_text>", "<|eot_id|>"],
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id    
#     )
#     output = tokenizer.decode(generation_output.sequences[0])
#     return output[0:len(message)], output[len(message):]
      
# def chat(message, history):
#     history = history or []

#     # TODO: Introduce full context into chat
#     message_formatted = llama3_format_prompt(message)

#     # TODO: Remove special tokens
#     _, response = sample_response(message_formatted)

#     history.append((message, response))
#     return history, history

def update(message, history):
    message = message.strip()
    # if len(message) != 0:   
    #     history.append((message, f"Welcome to Gradio, {message}!"))

    # else:
    #     raise gr.Error("Chat messages cannot be empty")
    
    # return "", history
    return "", history + [[message, None]]

def validate(message):
    if len(message) == 0:
        raise Exception()

    else:
        return True
    
def sample(history):
    # Build context string from history of conversation
    user_msgs = [llama3_format_prompt(m[0]) for m in history[:-1]]
    assistant_msgs = [llama3_format_response(m[1]) for m in history[:-1]]
    message = ''.join([x + y for x,y in zip(user_msgs, assistant_msgs)])
    message = message + llama3_format_prompt(history[-1][0])

    inputs = tokenizer(message, return_tensors="pt")["input_ids"].cuda()
    generation_kwargs = dict(
                            input_ids=inputs,
                            streamer=streamer,
                            # max_new_tokens=20,
                            generation_config=GenerationConfig(do_sample=True, temperature=1.0, top_p=0.75, num_beams=1),
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_new_tokens=512,
                            tokenizer=tokenizer,
                            stop_strings=["<end_of_text>", "<|eot_id|>"],
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    history[-1][1] = "" # f"{len(message)}\n\n"
    # stream_count = 0
    for new_text in streamer:
        # new_text = ''.join(tokenizer.tokenize(new_text, add_special_tokens=False))
        # stream_count += 1
        # if stream_count == 1:
        #     continue
        history[-1][1] += new_text
        # time.sleep(0.05)
        yield history

out = gr.Chatbot(
        height="80rem",
        bubble_full_width=False,
        avatar_images=['../test-gradio/user.svg', '../test-gradio/bot.svg'])

with gr.Blocks(theme='freddyaboulton/dracula_revamped') as demo:
    gr.Markdown("Start typing below and then click **Submit** to see the output.")

    with gr.Row():
        with gr.Column(scale=1):
            input = gr.Textbox(label='User', placeholder='Input', interactive=True)
            
            with gr.Row():
                with gr.Column(scale=1):
                    btn_submit = gr.Button("Submit")
                    
                with gr.Column(scale=1):
                    btn_clear = gr.ClearButton([input, out])
                    btn_clear.click(lambda: None, None, out, queue=False)
        
        with gr.Column(scale=1):
            out.render()

    btn_submit.click(validate, [input], []).success(update, inputs=[input, out], outputs=[input, out], queue=False).then(sample, out, out)
    input.submit(validate, [input], []).success(update, inputs=[input, out], outputs=[input, out], queue=False).then(sample, out, out)
    
    # with gr.Row():
        
    #     out = gr.Textbox()
    # btn = gr.Button("Run")
    # btn.click(fn=update, inputs=inp, outputs=out)
    
    # text = gr.Textbox(label="Enter text to write to log file")
    # with gr.Row():
    #     for l in ["debug", "info", "warning", "error", "critical"]:
    #         button = gr.Button(f"log as {l}")
    #         button.click(fn=create_log_handler(l), inputs=text)
    # Log(log_file, dark=False)


# ifc = gr.Interface(
#     chat,
#     ["text", "state"],
#     ["chatbot", "state"],
#     allow_flagging="never",
#     theme='freddyaboulton/dracula_revamped')

# ifc.launch()

if __name__ == "__main__":
     demo.queue()
     demo.launch()