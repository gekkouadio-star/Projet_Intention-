import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_youtube_comments(video_url, max_scrolls=10):
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Requis pour le cloud
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--mute-audio")
    
    driver = webdriver.Chrome(options=chrome_options)
    comments = []
    
    try:
        driver.get(video_url)
        wait = WebDriverWait(driver, 15)
        
        # Attendre le chargement initial et scroller pour activer les comms
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 600);")
        
        # Récupérer les commentaires par vagues
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        for _ in range(max_scrolls):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height: break
            last_height = new_height

        elements = driver.find_elements(By.CSS_SELECTOR, "#content-text")
        comments = [el.text for el in elements if el.text]
        
    finally:
        driver.quit()
    return comments