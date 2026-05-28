import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

# 1. 데이터셋 로드
# 해당 데이터셋은 'image'와 'label'(단수형) 컬럼을 가지며, train/test 스플릿만 있습니다.
dataset = load_dataset("yourdatasets")

# 라벨 정보 확인 및 매핑 딕셔너리 생성
labels = dataset["train"].features["label"].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}


# 2. 이미지 전처리 (Processor) 설정
model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)


# 이미지 변환 함수 정의 
def transforms(examples):
    inputs = image_processor(
        [img.convert("RGB") for img in examples["image"]], return_tensors="pt"
    )
    inputs["labels"] = examples["label"] 
    return inputs


# 데이터셋에 실시간 전처리 적용
dataset = dataset.with_transform(transforms)


# 배치를 만들기 위한 콜레이터(Collator) 함수
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

# 3. 모델 설정
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

#4. 정확도


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    # 정확도(Accuracy) 계산
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./vit-result",
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    eval_strategy="epoch",  
    save_strategy="epoch", 
    learning_rate=5e-5,
    num_train_epochs=3, 
    logging_steps=50,
    load_best_model_at_end=True, 
    metric_for_best_model="accuracy",  
    remove_unused_columns=False,  
    fp16=True,  
    dataloader_num_workers=2, 
    optim="adamw_torch_fused",
)

# 6. 트레이너(Trainer) 정의 및 학습 진행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  #여기
    eval_dataset=dataset["test"],
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# 학습 시작
print("학습을 시작")
trainer.train()

# 7. 정확도 측정

print("정확도 측정 중")
metrics = trainer.evaluate()

print(f"\n[최종 결과] 테스트 데이터셋 정확도: {metrics['eval_accuracy']:.4f}")