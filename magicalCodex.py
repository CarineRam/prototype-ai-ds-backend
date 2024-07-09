from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer, BertForMaskedLM, pipeline, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BertTokenizerFast
import os
import json
import torch
import re
from torch.utils.data import DataLoader, Dataset, random_split
from typing import List, Dict
from parameters import save_parameters, print_all_parameters

magicalCodex_blueprint = Blueprint('magicalCodex', __name__)

SELECTED_MODEL_MC = 'selected_model_MC.json'
DATASETS_MC_DIR = 'datasets_MC'

selected_dataset_mc = None
dataset_text = None



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
    max_length = data.get('max_length')
    temperature = data.get('temperature')
    stopSequences = data.get('stopSequences')

    # print(f"Input Text for Bert : {input_text}")

    prediction = bert_predict_mask(input_text)

    # save_parameters('Bert', temperature, max_length, stopSequences)
    # print_all_parameters()
    print("Prediction :", prediction)
    return jsonify({'predicted_text': prediction})

@magicalCodex_blueprint.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text')
    max_length = data.get('max_length', 65)
    temperature = data.get('temperature')
    stopSequences = data.get('stopSequences')

    # print(f"Received input for GPT-2 generation: {input_text}")

    # save_parameters('GPT-2', temperature, max_length, stopSequences) ###TEST SQL
    # print_all_parameters()
    generated_text = gpt2_generate_response(input_text, max_length)
    return jsonify({'generated_text': generated_text})


#prediction of gpt2
def gpt2_generate_response(input_text, max_length):

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = gpt2model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text[len(input_text):].strip()

#
@magicalCodex_blueprint.route('/save_parameters', methods=['POST'])
def save_parameters_route():
    data = request.json
    name = data.get('name')
    model = data.get('model')
    max_length = data.get('max_length', 65)
    temperature = data.get('temperature')
    stopSequences = data.get('stopSequences')

    save_parameters(name, model, temperature, max_length, stopSequences)
    print_all_parameters()

    return redirect(url_for('magicalCodex.view_parameters'))

@magicalCodex_blueprint.route('/view_parameters', methods=['GET'])
def view_parameters():
    rows = print_all_parameters()
    parameters = []

    for row in rows:
        parameters.append({
            'id': row[0],
            'name': row[1],
            'model': row[2],
            'temperature': row[3],
            'max_tokens': row[4],
            'stop_sequences': row[5],
            'save_date': row[6]
        })

    return jsonify(parameters) #WATCH THE PROBLEM WITH THE VISUALIZATION IN APPVARIANT PAGE
    # return render_template('view_parameters.html', rows=rows)

#dataset choice
@magicalCodex_blueprint.route('/datasets_MC', methods=['GET'])
def list_datasets_MC():
    datasets_MC = [f for f in os.listdir(DATASETS_MC_DIR) if f.endswith('.txt')]
    print ("datasets_MC", datasets_MC)
    return jsonify(datasetsMC=datasets_MC)

# @magicalCodex_blueprint.route('/process_dataset', methods=['POST'])
# def process_dataset():
#     dataset_name = request.json.get('dataset_name')
#     model_type = request.json.get('model_type')
#     dataset_path = os.path.join(DATASETS_MC_DIR, dataset_name)

#     if not os.path.exists(dataset_path):
#         return jsonify(error="Dataset not found"), 404
    
#     with open(dataset_path, 'r') as file:
#         dataset_content = file.read()

#         selected_model = load_selected_model_MC()

#         if not selected_model or selected_model not in models_MC:
#             return jsonify(error="Model not selected or invalid"), 400
        
#         if selected_model == 'berttokenizer':
#             bert_predict_mask(input_text=any)

#         elif selected_model == 'gpt2tokenizer':
#             gpt2_generate_response(input_text=any)

#         return jsonify(input=dataset_content)
    
# @magicalCodex_blueprint.route('/magicalCodex/process_selected_dataset', methods=['GET'])
# def process_selected_dataset():
#     global selected_dataset_mc
#     if selected_dataset_mc:
#         dataset_path = os.path.join(DATASETS_MC_DIR, selected_dataset_mc)
#         with open(dataset_path, 'r') as file:
#             data = file.readlines()

#         responses = [gpt2_generate_response(line.strip()) for line in data]

#         return jsonify({"status": "success", "message": f"Dataset {selected_dataset_mc} processed with GPT-2", "responses": responses})
#     else:
#         return jsonify({"status": "error", "message": "No dataset selected"}), 400

#Retrieve dataset data
def get_dataset():
    global dataset_name, dataset_text
    dataset_name = request.form['dataset_name']
    dataset_path = os.path.join(DATASETS_MC_DIR, dataset_name)

    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset_text = file.read()
        print(dataset_text[:100])

#Personnalized dataset
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.textDataset = []
        for i in range(0, len(text), block_size):
            tokens = tokenizer.encode(text[i:i + block_size])
            tokens_with_special_tokens = tokenizer.build_inputs_with_special_tokens(tokens)
            self.textDataset.append(tokens_with_special_tokens)

    def __len__(self):
        return len(self.textDataset)
    
    def __getitem__(self, item):
        # return torch.tensor(self.textDataset[item])
        return self.textDataset[item]

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {}
        # Assume 'input_ids', 'attention_mask', etc., are in each feature
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        # Add other necessary fields as required by your model
        return batch

#Initialization of the tokenizer and gpt2model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
modelgpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

#Train dataset GPT2
@magicalCodex_blueprint.route('/dtrain_gpt2', methods=['POST']) 
def gpt2_train_dataset():
    # global tokenizer, model
    data = request.get_json()
    dataset_name = data.get('dataset_name')
    model_name = data.get('model_name')
    epoch = data.get('epoch')

    epoch = float(epoch)

    if not dataset_name or not model_name:
        return jsonify({'error':'Missing dataset_name or model_name'}), 400

    dataset_path = os.path.join(DATASETS_MC_DIR, dataset_name)

    try:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            dataset_text = file.read()
            # print(dataset_text[:100])
    except FileNotFoundError:
        return jsonify({'error': f"Dataset {dataset_name} not found"}), 404

    cleaned_text = re.sub(r'[^\w\s]', '', dataset_text)
    cleaned_text = cleaned_text.lower()
    # print(cleaned_text)
    
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token':'[PAD]'})

    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    modelgpt2.resize_token_embeddings(len(tokenizer))

    block_size = 128
    dataset = TextDataset(cleaned_text, tokenizer, block_size)

    # print("dataset = TextDataset(cleaned_text ... )")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    training_args = TrainingArguments(
       output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch" 
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=modelgpt2,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    return jsonify({'message': 'Model GPT2 trained successfully'})
    
#Generate output GPT2 after training dataset
@magicalCodex_blueprint.route('/dgenerate_text', methods=['POST'])
def gpt2_generate_text():
    data = request.get_json()
    prompt_text = data.get('prompt_text')
    # print(prompt_text)

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    outputs = modelgpt2.generate(input_ids,attention_mask=attention_mask, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    # print(f"Model Ouputs: {outputs}")

    if outputs is not None and len(outputs) > 0:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Generated Text: {generated_text}")
        generated_part = generated_text[len(prompt_text):].strip()
        return jsonify({'generated_text': generated_part})
    else:
        return jsonify({'error':'Failed to generate text'}), 500


tokenizerBert = BertTokenizerFast.from_pretrained('bert-base-uncased')
modelBert = BertForMaskedLM.from_pretrained("bert-base-uncased")

class CustomTextDataset(Dataset):
    def __init__(self, tokenized_text):
        self.input_ids = tokenized_text['input_ids']
        self.attention_mask = tokenized_text['attention_mask']
        if 'token_type_ids' in tokenized_text:
            self.token_type_ids = tokenized_text['token_type_ids']
        else:
            self.token_type_ids = None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
        if self.token_type_ids is not None:
            item['token_type_ids'] = self.token_type_ids[idx]
        return item

#Train dataset Bert
@magicalCodex_blueprint.route('/dtrain_bert', methods=['POST'])
def bert_train_dataset():
    # global tokenizerBert, modelBert
    data = request.get_json()
    dataset_name = data.get('dataset_name')
    model_name = data.get('model_name')
    # epoch = data.get('epoch', 3)
    epoch = data.get('epoch')

    epoch = float(epoch)

    if not dataset_name or not model_name:
        return jsonify({'error':'Missing dataset_name or model_name'}), 400

    dataset_path = os.path.join(DATASETS_MC_DIR, dataset_name)

    try:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            dataset_text = file.read()
            # print(dataset_text[:100])
    except FileNotFoundError:
        return jsonify({'error': f"Dataset {dataset_name} not found"}), 404
    
    cleaned_text = re.sub(r'[^\w\s]', '', dataset_text)
    cleaned_text = cleaned_text.lower()
    # print(cleaned_text)
    lines = cleaned_text.split('\n')

    # block_size= 128
    # dataset = TextDataset(lines, tokenizerBert, block_size=block_size)

    tokenized_text = tokenizerBert(lines, padding=True, truncation=True, max_length=128, return_tensors='pt')

    train_size = int(0.9 * len(tokenized_text['input_ids']))
    val_size = len(tokenized_text['input_ids']) - train_size

    # train_dataset = Dataset.from_dict({key: value[:train_size] for key, value in tokenized_text.items()})
    # val_dataset = Dataset.from_dict({key: value[train_size:] for key, value in tokenized_text.items()})

    train_dataset = CustomTextDataset({key: value[:train_size] for key, value in tokenized_text.items()})
    val_dataset = CustomTextDataset({key: value[train_size:] for key, value in tokenized_text.items()})

    # train_size = int(0.9 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=epoch,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizerBert, mlm=False)

    trainer = Trainer(
        model=modelBert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    try:
        trainer.train()
    except Exception as e:
        return jsonify({'error':f"Training error: {str(e)}"}), 500

    return jsonify({'message': 'Model Bert trained successfully'})

#Predict MASK output Bert after training dataset
def bert_predict(prompt_text, tokenizer, model):
    modelBert.eval()

    input_ids = tokenizerBert(prompt_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**input_ids)
        predictions = outputs.logits

    masked_index = torch.where(input_ids["input_ids"] == tokenizerBert.mask_token_id)[1]
    predicted_token_id = predictions[0, masked_index].argmax(dim=-1).item()

    predicted_token = tokenizerBert._convert_id_to_token(predicted_token_id)
    # print(f"Predicted TOken: {predicted_token}")

    input_text_list = prompt_text.split()
    input_text_list[input_text_list.index('[MASK]')] = predicted_token
    completed_text = ' '.join(input_text_list)

    return completed_text

@magicalCodex_blueprint.route('/dpredict_word', methods=['POST'])
def bert_predict_word():
    # global tokenizerBert, modelBert
    data = request.get_json()
    prompt_text = data.get('prompt_text')

    if not prompt_text:
        return jsonify({'error': 'Missing text input'}), 400
    
    predicted_token = bert_predict(prompt_text, tokenizerBert, modelBert)
    return jsonify({'predicted_token': predicted_token})

#Save the parameters in the 