# 2023_OSSP_Google_Crawling

# Step 1. 필요한 모듈과 라이브러리를 로딩하고 검색어를 입력 받습니다
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import urllib.request
import urllib.parse
import time

# Step 2. 사용자에게 검색 관련 정보들을 입력 받습니다.
print("=" *100)
print(" 2023_OSSP_Google_Crawling ")
print("=" *100)
query_txt = input('1. 검색어를 입력하시오 : ')
file_name = input('1. 파일명을 설정하시오 : ')


# Step 3. 크롬 드라이버 설정 및 웹 페이지 열기
s = Service("C:/Users/rmsdu/OneDrive/문서/GitHub/2023_OSSP/chromedriver.exe")
driver = webdriver.Chrome(service=s)  # 구글 웹드라이버를 사용
driver.implicitly_wait(3)

url = "https://www.google.co.kr/imghp?hl=ko&ogbl" # 구글 이미지 페이지 주소
driver.get(url)  # 구글 이미지 페이지로 들어간다.
time.sleep(3)

# Step 4. 자동으로 검색어 입력 후 조회하기
element = driver.find_element(By.NAME, 'q')
element.send_keys(query_txt)
element.send_keys(Keys.RETURN)

# 스크롤 끝까지 내린다음에 사진 다운로드를 시작한다.
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

# find_element vs find_elements : 이미지를 여러개 선택해서 리스트로 넣을 수 있다.
# .rg_i.Q4LuWd : 다운받을 이미지들의 class 명으로 가져온다.
# [0].click() : 첫 번째 이미지를 클릭하겠다.

images = driver.find_elements(By.CSS_SELECTOR,'.rg_i.Q4LuWd')  # 검색해서 나온 이미지
count = 1
for image in images:
    try:
        image.click()
        time.sleep(3)
        imgUrl = driver.find_element(By.XPATH_SELECTOR,'/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")  # 다운받을 이미지 창 띄우기
        # src : 재생할 미디어 파일의 URL을 명시
        urllib.request.urlretrieve(imgUrl, file_name + str(count) + ".jpg")  # 지정한 파일이름으로 저장
        print(count, "download success")
        count += 1
    except:
        pass


driver.close() #브라우저 창을 닫아주기

print("crawling end")
driver.close()