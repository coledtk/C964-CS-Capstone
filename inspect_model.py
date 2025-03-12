import pickle

# Load and print the model structure
with open('sms_spam_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("Type of loaded object:", type(model_data))
if isinstance(model_data, dict):
    print("Keys in dictionary:", model_data.keys())
    if 'model' in model_data:
        print("Type of model:", type(model_data['model']))
        if hasattr(model_data['model'], 'named_steps'):
            print("Named steps:", model_data['model'].named_steps.keys())
    else:
        print("No 'model' key found in dictionary")
else:
    if hasattr(model_data, 'named_steps'):
        print("Named steps:", model_data.named_steps.keys())
    else:
        print("Loaded object does not have 'named_steps' attribute")