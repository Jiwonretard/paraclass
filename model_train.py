import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

# ==========================================
# 1. 데이터셋 로드 (허깅페이스에서 코드로 가져오기)
# ==========================================
# 'beans' 데이터셋은 'image' 컬럼과 'labels' 컬럼을 가지고 있습니다.
dataset = load_dataset("Donghyun99/Stanford-Cars")

# 라벨 정보 확인 및 매핑 딕셔너리 생성
labels = dataset["train"].features["labels"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

print(f"클래스 종류: {labels}")

# ==========================================
# 2. 이미지 전처리 (Processor) 설정
# ==========================================
# 사용할 모델 아키텍처에 맞는 이미지 프로세서를 불러옵니다.
model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)


# 이미지 변환 함수 정의 (RGB 변환 및 텐서화)
def transforms(examples):
    inputs = image_processor(
        [img.convert("RGB") for img in examples["image"]], return_tensors="pt"
    )
    inputs["labels"] = examples["labels"]
    return inputs


# 데이터셋에 실시간 전처리 적용
dataset = dataset.with_transform(transforms)


# 배치를 만들기 위한 콜레이터(Collator) 함수
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


# ==========================================
# 3. 모델 설정
# ==========================================
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# ==========================================
# 4. 평가지표 (정확도) 정의
# ==========================================


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    # 정확도(Accuracy) 계산
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


# ==========================================
# 5. 학습 인자(Training Arguments) 설정
# ==========================================
training_args = TrainingArguments(
    output_dir="./vit-beans-result",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",  # 매 에포크마다 평가 진행
    save_strategy="epoch",  # 매 에포크마다 모델 저장
    learning_rate=5e-5,
    num_train_epochs=3,  # 3에포크 학습
    logging_steps=10,
    load_best_model_at_end=True,  # 가장 성적이 좋은 모델을 최종 로드
    metric_for_best_model="accuracy",  # 가장 높은 정확도 모델 기준
    remove_unused_columns=False,  # 전처리된 컬럼 유지를 위해 필수
)

# ==========================================
# 6. 트레이너(Trainer) 정의 및 학습 진행
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],  # 검증 데이터셋
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# 학습 시작
print("--- 학습을 시작합니다 ---")
trainer.train()

# ==========================================
# 7. 최종 테스트 데이터셋으로 정확도 측정
# ==========================================
print("--- 최종 테스트 정확도 측정 중 ---")
metrics = trainer.evaluate(dataset["test"])

print(f"\n[최종 결과] 테스트 데이터셋 정확도(Accuracy): {metrics['eval_accuracy']:.4f}")