import re
import pandas as pd


def initial_preprocess(df):
    # 결측치 제거
    df.dropna(inplace=True) 

    # 중복값제거
    df.drop_duplicates(inplace=True) 

    # 날짜열 제거
    df = df[['TITLE', 'CONTENT']]
    
    # 인덱스 재정렬
    df.reset_index(drop=True, inplace=True)

    return df


def string_preprocess(df):
    # 마침표 기준 split
    df['TEMP1'] = 0
    for i in range(len(df['CONTENT'])):
        if df['CONTENT'][i].find('.') != -1:
            df['TEMP1'][i] = df['CONTENT'][i].split('.')
        else: 
            df['TEMP1'][i] = df['CONTENT'][i]

    # 사진 문자열 제거(맨 마지막)
    df['TEMP2'] = df['TEMP1'].apply(lambda x : x[:-1])

    # [스포탈코리아] 신문지 이름 제거
    for i in range(len(df['TEMP2'])):
        if '스포탈코리아' in df['TEMP2'][i][0]:
            df['TEMP2'][i][0] = ' '.join(df['TEMP2'][i][0].split(' ')[1:])
        else:
            pass

    # '*** 기자= ' 형식 제거
    for i in range(len(df['TEMP2'])):
        if df['TEMP2'][i][0].find('=') != -1:
            df['TEMP2'][i][0] = df['TEMP2'][i][0][df['TEMP2'][i][0].find('= ')+2:]

    # 마침표 기준 join
    for i in range(len(df['TEMP2'])):
        df['TEMP2'][i] = '. '.join(df['TEMP2'][i])

    # (한국 시간) 없애기
    for i in range(len(df['TEMP2'])):
        df['TEMP2'][i] = df['TEMP2'][i].replace("(한국 시간) or (한국시간)", "")
    
    return df


def regex_for_special(text):
    # E-mail제거
    email_pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' 
    text = re.sub(pattern=email_pattern, repl='', string=str(text))
    
    # 이미지 태그 제거
    text = re.sub("<img.*[0-9]", "", string=text)  
    text = re.sub('img src=.*JPG">', "", string=text)
    text = re.sub('. jpg\">|. JPG">', '', string=text)  
    
    # 과도한'\n' 제거
    text = re.sub('\n+', ' ', text)  
    
    # html 태그 제거
    ptag_pattern = re.compile('[</p>].*[<p>]', re.DOTALL)
    text = re.sub("<b>.*</b>", "", string=text)
    text = re.sub("<B>.*</B>", "", string=text) 
    text = re.sub('</p>\n<p>', "", text)
    remove = ptag_pattern.findall(text)
    for i in remove:
        text = text.replace(i, "")

    # 이모지 제거
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # 특수문자 제거
    text = re.sub("&#8211;", "", string=text) 
    text = re.sub("②=FOOT", "", string=text)
    text = re.sub(":.", "", string=text)
    text = re.sub("[→■▶▲○※↓]", "", string=text)

    # 기타 제거
    text = re.sub("(https://.*.kr/)",'', string = text)
    text = re.sub("&nbsp;|&#8211;|&#8729;|&#48419;","", string=text)
                                                                                                             
    return text


# 라벨 생성 전 최종 데이터프레임 형식
def pre_ground_truth(df):

    # 최종 text 구성 = title + content
    df['TITLE_CONT'] = df['TITLE'] + df['CLEANED']

    # 필요한 열만 선택
    df = df.iloc[:,[0,1,-1]]

    # summary열 생성
    df['TRUE SUMMARY'] = ''

    return df


# 전체 전처리 적용 함수
def preprocessing(df):

    # 기본 전처리 적용
    df = initial_preprocess(df)

    # 문자열 전처리 적용
    df = string_preprocess(df)

    # 특수문자, html tags, 몇가지 패턴 정규식 적용
    df['CLEANED'] = df['TEMP2'].apply(lambda x: regex_for_special(x))

    # 최종 데이터프레임 형식
    df = pre_ground_truth(df)

    return df