import os
from flask import Flask, render_template, request
from transformers import pipeline
import requests

# Set your Hugging Face LLaMA API key here
os.environ["HF_LLAMAFINETUNE_API_KEY"] = 'LA-4dec858a6c8246329116ccfacc50313047c021c491644fb0803801eb4e8b566b'  # Replace with your LLaMA API key

app = Flask(__name__)

# Define the Hugging Face LLaMA model using your LLaMA-specific API key
llm_resto = pipeline("text-generation", model="facebook/llama-7b", 
                     use_auth_token=os.getenv("HF_LLAMAFINETUNE_API_KEY"), device=0)  # Ensure CUDA is available if using device=0

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == "POST":
        # Collecting form data
        age = request.form['age']
        gender = request.form['gender']
        weight = request.form['weight']
        height = request.form['height']
        veg_or_noveg = request.form['veg_or_nonveg']
        disease = request.form['disease']
        region = request.form['region']
        allergics = request.form['allergics']
        foodtype = request.form['foodtype']

        # Construct the input prompt
        prompt = f"""
        Diet Recommendation System:
        I want you to recommend 6 restaurants names, 6 breakfast names, 5 dinner names, and 6 workout names,
        based on the following criteria:
        Person age: {age}
        Person gender: {gender}
        Person weight: {weight}
        Person height: {height}
        Person veg_or_nonveg: {veg_or_noveg}
        Person generic disease: {disease}
        Person region: {region}
        Person allergics: {allergics}
        Person foodtype: {foodtype}.
        """

        # Use the model to generate recommendations
        results = llm_resto(prompt, max_length=1000)[0]['generated_text']

        # Extracting the different recommendations using regular expressions
        import re

        restaurant_names = re.findall(r'Restaurants:(.*?)Breakfast:', results, re.DOTALL)
        breakfast_names = re.findall(r'Breakfast:(.*?)Dinner:', results, re.DOTALL)
        dinner_names = re.findall(r'Dinner:(.*?)Workouts:', results, re.DOTALL)
        workout_names = re.findall(r'Workouts:(.*?)$', results, re.DOTALL)

        # Cleaning up the extracted lists
        restaurant_names = [name.strip() for name in restaurant_names[0].strip().split('\n') if name.strip()]
        breakfast_names = [name.strip() for name in breakfast_names[0].strip().split('\n') if name.strip()]
        dinner_names = [name.strip() for name in dinner_names[0].strip().split('\n') if name.strip()]
        workout_names = [name.strip() for name in workout_names[0].strip().split('\n') if name.strip()]

        return render_template('result.html', restaurant_names=restaurant_names, 
                               breakfast_names=breakfast_names, dinner_names=dinner_names, 
                               workout_names=workout_names)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
