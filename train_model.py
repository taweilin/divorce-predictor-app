import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("divorce.csv", sep=';')

questions = ['如果我們的爭論惡化，只要其中一人道歉，爭論就會結束。', '即使關係變得困難，我們也知道可以忽略彼此的差異。', '當有需要時，我們可以從頭開始並修正討論。', '當我與伴侶溝通時，最終會達成共識。', '我與伴侶共度的時光對我們來說是特別的。', '我們在家中沒有當作伴侶相處的時間。', '我們就像住在同一個屋簷下的陌生人，而不是家人。', '我喜歡和伴侶一起度假。', '我喜歡和伴侶一起旅行。', '我和伴侶對婚姻角色有相似看法。', '我和伴侶對信任有共同價值觀。', '我了解伴侶基本的焦慮來源。', '我知道伴侶目前的壓力是什麼。', '我認識伴侶的朋友和他們的社交關係。', '我覺得我很了解我的伴侶。', '我了解伴侶的內心世界。', '我會與伴侶分享親密的事情。', '伴侶也很了解我。', '伴侶知道我基本的焦慮來源。', '伴侶知道我目前的壓力。', '我覺得伴侶不了解我。', '我知道伴侶最喜歡的食物。', '我知道伴侶最喜歡的電影。', '我知道伴侶最喜歡的顏色。', '我知道伴侶最好的朋友是誰。', '我知道伴侶想成為什麼樣的人。', '我知道伴侶現在是什麼樣的人。', '我覺得伴侶知道我真正的想法。', '當我們爭吵時，我覺得我受到個人攻擊。', '我們爭吵時，比較像在打架而不是討論。', '我們無法在不傷害對方的情況下解決衝突。', '我們在討論時會立刻情緒化。', '我常常在爭論中大喊大叫。', '我無法控制自己在爭吵時大喊。', '爭吵時我會辱罵我的伴侶。', '我通常會對伴侶展現親密的肢體接觸。', '我有時會故意沉默來傷害伴侶。', '我在討論中有時會使用暴力。', '我在討論中會說髒話。', '我會在爭吵中威脅我的伴侶。', '我在討論中會提高音量。', '我會在討論結束前離開現場。', '我會盡快讓自己冷靜下來。', '我知道如何讓自己冷靜下來。', '我會在討論中試著讓伴侶冷靜下來。', '我相信我們可以解決問題。', '我相信我們能克服差異。', '我和伴侶之間的溝通是健康的。', '我相信我們可以維持健康的溝通。', '我們可以在不爭吵的情況下討論問題。', '我相信我們的未來是有希望的。', '我相信我們是合得來的。', '我覺得我們的關係是穩固的。', '我覺得我在關係中的所作所為是正確的。']
df.columns = questions + ["Class"]

X = df.drop("Class", axis=1)
y = df["Class"].apply(lambda x: 1 - x)  # ← 重點：反轉標籤方向

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "divorce_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "question_list.pkl")