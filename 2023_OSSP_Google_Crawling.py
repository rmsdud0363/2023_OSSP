# 2023_OSSP_Google_Crawling

from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import urllib.request
import urllib.parse
import time

# Step 2. 사용자에게 검색 관련 정보들을 입력 받습니다.
print("=" *100)
print(" 2023_OSSP_Google_Crawling ")
print("=" *100)
query_txt = input('1. 검색어를 입력하시오 : ')
file_name = input('1. 파일명을 설정하시오 : ')

s = Service(r"C:\Users\rmsdu\OneDrive\문서\GitHub\2023_OSSP\2023_OSSP_Data\Train\chromedriver.exe")
driver = webdriver.Chrome(service=s)  # 구글 웹드라이버를 사용
driver.implicitly_wait(3)

url = "https://www.google.co.kr/imghp?hl=ko&ogbl" # 구글 이미지 페이지 주소
driver.get(url)  # 구글 이미지 페이지로 들어간다.
time.sleep(3)

element = driver.find_element(By.NAME, 'q')
element.send_keys(query_txt)
element.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")  # 자바스크립터 실행, 브라우저의 높이를 저장한다.
while True:  # 무한반복
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 브라우저 끝까지 스크롤을 내리겠다.

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)  # 로딩될 때를 기다린다.

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:  # 끝까지 내려진 상황
        try:
            driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input').click()  # 결과 더보기 선택
        except:  # 결과 더보기 칸이 없을 때
            break
    last_height = new_height


images = driver.find_elements(By.CSS_SELECTOR,'.rg_i.Q4LuWd')
links = []
for image in images:
    if image.get_attribute('src')!=None:
        links.append(image.get_attribute('src'))

print(query_txt + '찾은 이미지 개수: ', len(links))
time.sleep(2)

for k, i in(enumerate(links[:15], start=1)):
    url = i
    start = time.time()
    urllib.request.urlretrieve(url, file_name + str(k) + ".jpg")  # 지정한 파일이름으로 저장

print(query_txt + '--다운 완료--')
driver.close()