import requests
import os

def fetch_sample_biometrics():
    # Direct reliable links to public domain ear images for biometric testing
    samples = {
        "Sample_Person_A": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Human_ear.jpg",
        "Sample_Person_B": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Left_ear_of_a_man.jpg/640px-Left_ear_of_a_man.jpg",
        "Sample_Person_C": "https://upload.wikimedia.org/wikipedia/commons/b/b3/Right_Ear_of_a_Human.jpg"
    }
    
    save_dir = "data/biometric_samples"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Fetching verified biometric samples for Ear Recognition...")
    
    for name, url in samples.items():
        try:
            response = requests.get(url, timeout=20, stream=True)
            if response.status_code == 200:
                filepath = os.path.join(save_dir, f"{name}.jpg")
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully fetched: {name}.jpg")
            else:
                print(f"Failed to fetch {name} (Status {response.status_code})")
        except Exception as e:
            print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    fetch_sample_biometrics()
