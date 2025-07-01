import pandas as pd


def compute_video_subscriber_contributions(
    ch_df: pd.DataFrame,
    result_L: pd.DataFrame,
    daily_avg: float,
    correction: float = 0.8,
    max_days: int = 14
) -> pd.DataFrame:
    """
    채널 일일 평균 구독자 증가량(daily_avg)을
    롱폼 영상의 "실제 조회수 증분(actual_delta) vs 기대 조회수 증분(expected_delta)" 차이에
    비례하여 분배한 뒤, 공개 2주(max_days) 동안 누적 구독자 기여량을 계산합니다.

    Parameters:
    - ch_df: DataFrame with ['timestamp','video_id','view_count','day_since_pub','is_short','subscriber_count']
    - result_L: DataFrame with ['day','avg_view_count']
    - daily_avg: 하루 평균 구독자 증가량 (일별 고정)
    - correction: 기대 조회수 증분 보정치 (곱할 계수)
    - max_days: 영상 기여 영향 기간 (일)

    Returns:
    - subs_df: DataFrame ['video_id','subs_contrib']
      각 영상 누적 구독자 기여량
    """
    # 1) 롱폼 영상(is_short=False) 및 max_days 이내 필터링
    df = ch_df[(ch_df['is_short'] == False) & (ch_df['day_since_pub'] < max_days)].copy()

    # 2) 영상별·일자별 최대 view_count 집계
    daily = (
        df.groupby(['video_id', 'day_since_pub'])['view_count']
          .max()
          .reset_index()
    )

    # 3) 실제 조회수 증분(actual_delta) 계산
    daily = daily.sort_values(['video_id', 'day_since_pub'])
    daily['prev_views'] = (
        daily.groupby('video_id')['view_count']
             .shift(1)
             .fillna(0)
    )
    daily['actual_delta'] = daily['view_count'] - daily['prev_views']

    # 4) 기대 조회수 증분(expected_delta) 계산
    exp_map = result_L.set_index('day')['avg_view_count'].to_dict()
    daily['expected'] = daily['day_since_pub'].map(exp_map).fillna(0)
    daily['prev_expected'] = (
        daily.groupby('video_id')['expected']
             .shift(1)
             .fillna(0)
    )
    daily['expected_delta'] = daily['expected'] - daily['prev_expected']

    # 5) 보정된 증분(diff) = actual_delta - correction * expected_delta (음수는 0)
    daily['diff'] = (
        daily['actual_delta'] - correction * daily['expected_delta']
    ).clip(lower=0)

    # 6) 일자별 diff 총합 계산
    totals = (
        daily.groupby('day_since_pub')['diff']
             .sum()
             .rename('total_diff')
    )
    daily = daily.join(totals, on='day_since_pub')

    # 7) 일자별 구독자 분배(subs) = diff/total_diff * daily_avg
    daily['subs'] = daily.apply(
        lambda r: (r['diff'] / r['total_diff'] * daily_avg) if r['total_diff'] > 0 else 0,
        axis=1
    )

    # 8) 영상별 누적 구독자 기여량(subs_contrib) 계산
    subs_df = (
        daily.groupby('video_id')['subs']
             .sum()
             .reset_index()
             .rename(columns={'subs':'subs_contrib'})
    )

    return subs_df
