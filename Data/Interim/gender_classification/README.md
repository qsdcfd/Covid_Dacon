
## gender column에 포함된 other를 음성데이터를 통해 female, male로 분류

-6/20- => 비단 음성(mfcc)만을 이용해서 분류할께 아니라 fever여부, 통증 여부가 mfcc에 영향을 준다고 생각하므로 해당 column을 학습에 포함해서 gender의 other column을 분류해보자.

=> mfcc에 포함되는 이상치에대해 파악해보고 제거 후 학습을 진행해보자.
