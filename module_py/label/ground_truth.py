from pororo import Pororo

def ground_truth(df):

    # pororo abstractive 객체 생성
    pororo_abs = Pororo(task="summarization", model="abstractive", lang="ko")

    # ground truth 생성 (생성적 요약)
    df['TRUE SUMMARY'] = df['TITLE_CONT'].progress_apply(lambda x:pororo_abs(x))

    return df