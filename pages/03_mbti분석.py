import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
df = pd.read_csv('countriesMBTI_16types.csv')
print("데이터 형태:", df.shape)

# 2. 전 세계 평균 MBTI 분포 계산
mbti_cols = [c for c in df.columns if c != 'Country']
mean_mbti = df[mbti_cols].mean().sort_values(ascending=False)
print("\n[전 세계 평균 MBTI 유형별 비율]")
print(mean_mbti * 100)

# 3. 4가지 성향 차원별(E/I, N/S, T/F, J/P) 글로벌 평균 계산
E_types = [c for c in mbti_cols if c.startswith('E')]
I_types = [c for c in mbti_cols if c.startswith('I')]
N_types = [c for c in mbti_cols if c[1] == 'N']
S_types = [c for c in mbti_cols if c[1] == 'S']
T_types = [c for c in mbti_cols if c[2] == 'T']
F_types = [c for c in mbti_cols if c[2] == 'F']
J_types = [c for c in mbti_cols if c.endswith('J')]
P_types = [c for c in mbti_cols if c.endswith('P')]

df['Introversion'] = df[I_types].sum(axis=1)
df['Extraversion'] = df[E_types].sum(axis=1)
df['Intuition'] = df[N_types].sum(axis=1)
df['Sensing'] = df[S_types].sum(axis=1)
df['Thinking'] = df[T_types].sum(axis=1)
df['Feeling'] = df[F_types].sum(axis=1)
df['Judging'] = df[J_types].sum(axis=1)
df['Perceiving'] = df[P_types].sum(axis=1)

print("\n[성향 차원별 글로벌 평균]")
print(f"내향형(I): {df['Introversion'].mean()*100:.2f}% vs 외향형(E): {df['Extraversion'].mean()*100:.2f}%")
print(f"직관형(N): {df['Intuition'].mean()*100:.2f}% vs 감각형(S): {df['Sensing'].mean()*100:.2f}%")
print(f"사고형(T): {df['Thinking'].mean()*100:.2f}% vs 감정형(F): {df['Feeling'].mean()*100:.2f}%")
print(f"판단형(J): {df['Judging'].mean()*100:.2f}% vs 인식형(P): {df['Perceiving'].mean()*100:.2f}%")

# 4. 대한민국(South Korea) 데이터 조회
korea_data = df[df['Country'] == 'South Korea']
if not korea_data.empty:
    print("\n[대한민국 MBTI 분포 순위]")
    sk_sorted = korea_data[mbti_cols].iloc[0].sort_values(ascending=False) * 100
    print(sk_sorted)

# 5. 시각화 (상위 10개 유형 바차트)
plt.figure(figsize=(10, 6))
mean_mbti.head(10).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Global Average Top 10 MBTI Types', fontsize=14)
plt.xlabel('MBTI Type', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('global_mbti_top10.png')
print("\n시각화 파일('global_mbti_top10.png')이 저장되었습니다.")
