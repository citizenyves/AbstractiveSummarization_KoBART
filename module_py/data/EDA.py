import os, sys
import matplotlib.pyplot as plt


# 절대 경로 참조
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils import config, set_seed

# set seed
set_seed(config)

# tokenizer
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)

# aihub dataset + train(val 나누기 전) 문장 토큰수 확인
texts = [data['text'] for data in train_total]
sums = [data['summary'] for data in train_total]

len_text = []
len_sum = []

for text, sum in tqdm(zip(texts, sums), total=len(texts)):
    len_text.append(len(tokenizer.encode(text)))
    len_sum.append(len(tokenizer.encode(sum)))


# 문장 길이 통계치 확인
describe = pd.DataFrame(data={'text':len_text, 'summary':len_sum})
describe.describe(percentiles=[.05, .10, .25, .5, .75, .90, .95, .99, .999, .9999,.99999])


# 시각화
plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(figsize=(10, 5), facecolor="white", dpi=72) 
x = len_text
y = len_sum

x_bins = np.linspace(min(x), max(x), 140)
y_bins = np.linspace(min(y), max(y), 50)  

plt.hist2d(x, y, bins=[x_bins, y_bins], cmap="PiYG", norm=matplotlib.colors.LogNorm()) 
cbar = plt.colorbar() 
cbar.set_label("Counts", fontsize=14)

plt.xlim([0, 2500])
plt.ylim([0, 300])
plt.xlabel("Length of Text", fontsize=14)
plt.ylabel("Length of Summary", fontsize=14)

ax.tick_params(axis="both", direction="out")  
plt.grid(True, lw=0.15)

plt.tight_layout()  
# plt.savefig("2.png", dpi=300)
plt.show();







# trainset text에서 길이가 1024 넘는 샘플의 개수
text_over_1024 = [len for len in len_text if len > 1024]
summ_over_150 = [len for len in len_sum if len > 150]

print(f"train 데이터에서 길이 1024 이상 text 수 : {len(text_over_1024)}")
print(f"전체 train 데이터 샘플 수               : {len(train_total)}")
print(f"길이 1024 이상 text의 비중              : {round((len(text_over_1024) / len(train_total)) * 100, 4)}%")
print("\n")
print(f"train 데이터에서 길이 150 이상 summary 수 : {len(summ_over_150)}")
print(f"전체 train 데이터 샘플 수                 : {len(train_total)}")
print(f"길이 150 이상 summary 비중                : {round((len(summ_over_150) / len(train_total)) * 100, 4)}%")
