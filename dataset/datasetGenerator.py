import subprocess
import json
import os
import requests

# Define your prompts
nocode_prompt = """
Generate a list of 20 examples. For each example, include a "text" and a "label" field. 
The "label" should be either "noCode" or "containsCode".

- "noCode" examples should be short, natural-sounding sentences or paragraphs without any programming code or technical syntax.  

Output the result in JSON format, with each item like this:
{ "text": "...", "label": "..." }

IMPORTANT: Only output the JSON array, nothing else. No thinking process, no explanations.
"""

containsCode_prompt = """
Generate a list of 20 examples. For each example, include a "text" and a "label" field. 
The "label" should be either "noCode" or "containsCode".

- "containsCode" examples should be short, natural-sounding sentences or paragraphs with some programming code or technical syntax.  

Output the result in JSON format, with each item like this:
{ "text": "...", "label": "..." }

IMPORTANT: Only output the JSON array, nothing else. No thinking process, no explanations.
"""

def clean_json_response(text):
    # Find the first '[' and last ']'
    start = text.find('[')
    end = text.rfind(']') + 1
    
    if start == -1 or end == 0:
        return None
        
    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def generate_examples(prompt):
    try:
        print("Generating examples...")
        response = requests.post('http://localhost:11434/api/generate',
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False,
                "raw": True  # This helps get cleaner output
            }
        )
        
        if response.status_code == 200:
            print("Response received and code is: ", response.status_code)
            content = response.json()['response']
            
            # Clean and parse the JSON response
            examples = clean_json_response(content)
            if examples:
                return examples
            else:
                print("Error: Could not parse JSON response")
                print("Raw response:", content)
                return None
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error generating examples: {str(e)}")
        return None

def main():
    print("Starting dataset generation...")
    # Generate both types of examples
    nocode_examples = generate_examples(nocode_prompt)
    code_examples = generate_examples(containsCode_prompt)
    
    # Combine the examples
    dataset = []
    if nocode_examples:
        dataset.extend(nocode_examples)
    if code_examples:
        dataset.extend(code_examples)
    
    # Save to file
    with open("generated_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} examples and saved to generated_dataset.json")

if __name__ == "__main__":
    main()
