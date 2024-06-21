from flask import Blueprint, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer, BertForMaskedLM, pipeline, BertForSequenceClassification
import os
import json
import torch

magicalCodex_blueprint = Blueprint('magicalCodex', __name__)

SELECTED_MODEL_MC = 'selected_model_MC.json'
DATASETS_MC_DIR = 'datasets_MC'

selected_dataset_mc = None

gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2model = GPT2LMHeadModel.from_pretrained('gpt2')

berttokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bertmodel = BertForMaskedLM.from_pretrained("bert-base-uncased")

models_MC = {
    "gpt2tokenizer":{"GPT-2 is a large-scale language model developed by OpenAI, known for generating human-like text based on the input it receives. It uses a transformer architecture with 1.5 billion parameters, making it capable of performing various natural language processing tasks such as translation, summarization, and text generation. Despite its capabilities, GPT-2 also raised ethical concerns about the potential misuse of AI for generating misleading or harmful content."},
    "berttokenizer":{"BERT (Bidirectional Encoder Representations from Transformers) is a language model developed by Google that excels at understanding the context of words in a sentence by looking at both the preceding and following words. It utilizes a transformer architecture and has significantly improved the performance on various natural language processing tasks, such as question answering and language inference. BERT's bidirectional training approach allows it to capture the nuanced meaning and relationships in text more effectively than previous unidirectional models."},
}



# def bert_generate_text(input_text):
#     input_ids = berttokenizer.encode(input_text, return_tensors="pt")
#     output = bertmodel.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=berttokenizer.eos_token_id)
#     generated_text = berttokenizer.decode(output[0], skip_special_tokens=True)
#     return generated_text

# def gpt2_generate_response(input_text):
#     input_ids = gpt2tokenizer.encode(input_text, return_tensors="pt")
#     output = gpt2model.generate(input_ids, max_length=100, num_return_sequences=1)
#     generated_text = gpt2tokenizer.decode(output[0], skip_special_tokens=True)
#     return generated_text


if not os.path.exists(SELECTED_MODEL_MC):
    with open(SELECTED_MODEL_MC, 'w') as f:
        json.dump({}, f)

@magicalCodex_blueprint.route('/models_MC', methods=['GET'])
def get_models_MC():
    return jsonify({'models_MC' : list(models_MC.keys())})

#model choice
def save_selected_model_MC(model_MC_name, SELECTED_MODEL_MC):
    with open(SELECTED_MODEL_MC, 'w') as f:
        json.dump({'modelMC': model_MC_name}, f)

def load_selected_model_MC():
    if os.path.exists(SELECTED_MODEL_MC):
        with open(SELECTED_MODEL_MC, 'r') as f:
            return json.load(f).get('model_MC', '')
    return ''

@magicalCodex_blueprint.route('/select_models_MC', methods=['POST'])
def select_models_MC():
    data = request.json
    print("Data for model:", data)
    selected_model_MC = data.get('models_MC')

    if not selected_model_MC:
        return jsonify({'error':'No model provided'}), 400
    
    save_selected_model_MC(selected_model_MC)
    print("The selected model Magical Codex:", selected_model_MC)
    return jsonify({'success': True, 'selected_model_MC': selected_model_MC})

#prediction of bert
def bert_predict_mask(input_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bertmodel = BertForMaskedLM.from_pretrained('bert-base-uncased')

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        output = bertmodel(input_ids)

    mask_token_logits =  output.logits[0, mask_token_index, :]
    top_token_id = torch.argmax(mask_token_logits, dim=1)

    predicted_token = tokenizer.decode(top_token_id)

    input_text_list = input_text.split()
    input_text_list[input_text_list.index('[MASK]')] = predicted_token
    completed_text = ' '.join(input_text_list)

    return completed_text

@magicalCodex_blueprint.route('/predict_mask', methods=['POST'])
def predict_mask():
    data = request.json
    input_text = data.get('input_text')

    print(f"Input Text for Bert : {input_text}")

    prediction = bert_predict_mask(input_text)
    print("Prediction :", prediction)
    return jsonify({'predicted_text': prediction})

#prediction of gpt2
def gpt2_generate_response(input_text):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2model = GPT2LMHeadModel.from_pretrained('gpt2')

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = gpt2model.generate(input_ids, max_length=65, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text[len(input_text):].strip()


@magicalCodex_blueprint.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text')

    print(f"Received input for GPT-2 generation: {input_text}")

    generated_text = gpt2_generate_response(input_text)
    return jsonify({'generated_text': generated_text})

#dataset choice
@magicalCodex_blueprint.route('/datasets_MC', methods=['GET'])
def list_datasets_MC():
    datasets_MC = [f for f in os.listdir(DATASETS_MC_DIR) if f.endswith('.txt')]
    print ("datasets_MC", datasets_MC)
    return jsonify(datasetsMC=datasets_MC)

@magicalCodex_blueprint.route('/process_dataset', methods=['POST'])
def process_dataset():
    dataset_name = request.json.get('dataset_name')
    model_type = request.json.get('model_type')
    dataset_path = os.path.join(DATASETS_MC_DIR, dataset_name)

    if not os.path.exists(dataset_path):
        return jsonify(error="Dataset not found"), 404
    
    with open(dataset_path, 'r') as file:
        dataset_content = file.read()

        selected_model = load_selected_model_MC()

        if not selected_model or selected_model not in models_MC:
            return jsonify(error="Model not selected or invalid"), 400
        
        if selected_model == 'berttokenizer':
            bert_predict_mask(input_text=any)

        elif selected_model == 'gpt2tokenizer':
            gpt2_generate_response(input_text=any)

        return jsonify(input=dataset_content)
    
@magicalCodex_blueprint.route('/magicalCodex/process_selected_dataset', methods=['GET'])
def process_selected_dataset():
    global selected_dataset_mc
    if selected_dataset_mc:
        dataset_path = os.path.join(DATASETS_MC_DIR, selected_dataset_mc)
        with open(dataset_path, 'r') as file:
            data = file.readlines()

        responses = [gpt2_generate_response(line.strip()) for line in data]

        return jsonify({"status": "success", "message": f"Dataset {selected_dataset_mc} processed with GPT-2", "responses": responses})
    else:
        return jsonify({"status": "error", "message": "No dataset selected"}), 400




    
