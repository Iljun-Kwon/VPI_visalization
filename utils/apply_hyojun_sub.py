import os
import json
import datetime
import pandas as pd
from utils.subscriber_contrib import compute_video_subscriber_contributions
from utils.daily_contrib import compute_daily_video_subscriber_contributions_for_day

# 파일 경로 설정
STATUS_FILE = 'subs_status.json'
SUBS_FILE   = 'subs_contrib.csv'

# 상태 로드 함수: 마지막 처리 일자를 반환
def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE,'r') as f:
            data = json.load(f)
            return datetime.date.fromisoformat(data.get('last_run_date'))
    return None

# 상태 저장 함수: 오늘 날짜를 기록
def save_status(date: datetime.date):
    with open(STATUS_FILE,'w') as f:
        json.dump({'last_run_date': date.isoformat()}, f)

# 기존 누적 기여량 불러오기 (CSV->dict)
def load_subs():
    if os.path.exists(SUBS_FILE):
        df = pd.read_csv(SUBS_FILE)
        return dict(zip(df['video_id'], df['subs_contrib']))
    return {}

# 기여량 저장 (dict->CSV)
def save_subs(subs_dict):
    df = pd.DataFrame([{'video_id': vid, 'subs_contrib': cnt} for vid,cnt in subs_dict.items()])
    df.to_csv(SUBS_FILE, index=False)

# 1) 초기 배치: 과거 14일치 통합 계산
def initial_batch(ch_df, result_L):
    # 1-1) 지난 14일치 subscriber_delta 계산
    df_sorted = ch_df.sort_values('timestamp')
    cutoff = df_sorted['timestamp'].max() - pd.Timedelta(days=14)
    recent = df_sorted[df_sorted['timestamp'] >= cutoff]
    start_subs = recent['subscriber_count'].iloc[0]
    end_subs   = recent['subscriber_count'].iloc[-1]
    daily_avg  = (end_subs - start_subs) / 14

    # 1-2) 14일치 누적 기여량 계산
    subs_df = compute_video_subscriber_contributions(
        ch_df      = ch_df,
        result_L   = result_L,
        daily_avg  = daily_avg,
        correction = 0.8,
        max_days   = 14
    )
    # 1-3) 파일 저장 및 상태 기록
    subs_dict = dict(zip(subs_df['video_id'], subs_df['subs_contrib']))
    save_subs(subs_dict)
    save_status(datetime.date.today())
    print('Initial batch completed.')

# 2) 일일 업데이트: 오늘치만 계산 후 누적
def incremental_update(ch_df, result_L):
    today = ch_df['timestamp'].dt.date.max()
    last  = load_status()
    # 스킵 조건: 이미 오늘 처리됨
    if last == today:
        print(f'Skipping: already processed {today}')
        return

    # 2-1) 오늘 하루 subscriber 증가량 계산
    df_sorted = ch_df.sort_values('timestamp')
    day_df = df_sorted[df_sorted['timestamp'].dt.date == today]
    if day_df.empty:
        print('No data for today, skip')
        return
    s0 = day_df['subscriber_count'].iloc[0]
    s1 = day_df['subscriber_count'].iloc[-1]
    daily_delta = s1 - s0

    # 2-2) 오늘치 영상별 기여량 계산 함수 호출
    daily_subs_df = compute_daily_video_subscriber_contributions_for_day(
        ch_df      = ch_df,
        result_L   = result_L,
        date       = today,
        daily_delta= daily_delta,
        correction = 0.8,
        max_days   = 14
    )

    # 2-3) 이전 누적 불러와 합산
    prev = load_subs()  # dict video_id->subs_contrib
    for row in daily_subs_df.itertuples(index=False):
        prev[row.video_id] = prev.get(row.video_id, 0) + row.subs_contrib

    # 2-4) 저장 및 상태 기록
    save_subs(prev)
    save_status(today)
    print(f'Incremental update for {today} completed.')

# 스크립트 실행 진입점
if __name__ == '__main__':
    # ch_df, result_L 불러오는 로직 필요
    # 예: ch_df = pd.read_csv('channel_data.csv', parse_dates=['timestamp','published_at'])
    #     result_L = pd.read_csv('result_longform.csv')
    
    ch_df   = ...  # 사용자 환경에 맞게 로드
    result_L= ...  # 기대 조회수 데이터 로드

    # 상태가 없으면 최초 배치
    if load_status() is None:
        initial_batch(ch_df, result_L)
    else:
        incremental_update(ch_df, result_L)
