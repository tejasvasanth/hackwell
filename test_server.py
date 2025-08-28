import requests
import time

def test_server():
    try:
        response = requests.get("http://localhost:8000/docs")
        print(f"Server is running! Status code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("Server is not accessible")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing server accessibility...")
    for i in range(3):
        if test_server():
            break
        print(f"Attempt {i+1} failed, waiting 2 seconds...")
        time.sleep(2)
    else:
        print("Server is not accessible after 3 attempts")