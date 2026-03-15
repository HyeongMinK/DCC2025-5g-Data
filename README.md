# 2025 데이터 크리에이터 캠프 – 5G 데이터팀

> **수상: 우수상**

위성 영상을 활용한 산업단지 탐지 및 굴뚝 높이 추정 프로젝트입니다.
ESG 맥락에서 대기오염 모니터링을 위해 굴뚝의 위치·높이를 자동으로 분석하고,
멀티모달 환경 데이터를 융합해 산업단지 영역을 정밀하게 분할합니다.

---

## 폴더 구조

```
데이터크리에이터캠프_정리/
├── 5G데이터_예선제출물/                  # 예선 제출물
│   ├── mission1/                    # Mission 1: 굴뚝 탐지
│   │   ├── 5G데이터_mission1.ipynb
│   │   └── requirements.txt
│   ├── mission2/                    # Mission 2: 굴뚝 높이 회귀
│   │   ├── 5G데이터_mission2.ipynb
│   │   └── requirements.txt
│   └── mission3/                    # Mission 3: 산업단지 이진 분할
│       ├── 5G데이터_mission3.ipynb
│       └── requirements.txt
└── 5G데이터_본선제출물/               # 본선 제출물
    ├── 전이학습 모델/                 # Mission 4 사전학습
    │   └── pretrain_mit5.ipynb
    └── 전체 모델/                    # Mission 4 멀티모달 융합
        └── mission4_final.ipynb
```

---

## Mission 1 – 굴뚝 바운딩박스 탐지

**데이터:** KOMPSAT-3/3A 위성영상 (512×512 crops)
**모델:** YOLOv8x

### 모델 구조
- JSON polygon 라벨 → YOLO txt 형식 변환
- 학습 설정: epochs=200, imgsz=512, batch=32, seed=42

### 결과

| 지표 | 값 |
|------|-----|
| mAP50 | **0.993** |
| mAP50-95 | **0.905** |

---

## Mission 2 – 굴뚝 높이 회귀

**데이터:** KOMPSAT-3/3A 위성영상 + Polyline 라벨
**모델:** ConvNeXt-Tiny + ROIAlign + Cross-Attention

### 모델 구조
```
ConvNeXt-Tiny backbone
  └─ Feature map (768ch × 16×16)
       └─ ROIAlign (7×7) → 49개 ROI 토큰
            └─ Cross-Attention (3층)
                 ├─ Key/Value: ROI 토큰
                 └─ Query: 9D Geometry [dx, dy, sinθ, cosθ, cx, cy, side, dcx, dcy]
                      └─ Regression head → 굴뚝 높이 (m)
```

### 주요 설계 결정
- ROI가 이미지 경계를 벗어날 경우 **center-shift** 방식 처리 (clamp 대신 geometry 왜곡 방지)
- 높이 분포 불균형 → **WeightedRandomSampler** (log1p 변환 후 5분위 오버샘플링)
- 증강: HFlip(p=0.5), Brightness ±15%, VFlip 없음 (North-Up 이미지)

### 결과

| 지표 | 값 |
|------|-----|
| val RMSE | **4.939 m** |
| val MAE | **2.741 m** |
| MAPE | **2.90%** |

---

## Mission 3 – 산업단지 이진 분할

**데이터:** Sentinel-2 RGBN 위성영상 (512×512 TIF)
**모델:** SegFormer MiT-B5

### 모델 구조
```
4ch RGBN 입력
  └─ 1×1 Conv Adapter (4ch → 3ch)
       ├─ RGB 채널: Identity 초기화 (1.0)
       └─ NIR 채널: Zero 초기화 (학습으로 점진적 습득)
            └─ MiT-B5 (ImageNet pretrained)
                 └─ SegformerHead (256ch, Binary)
```

### 주요 설계 결정
- 라벨 변환(10→1, 90→0)은 `RemapLabels` 커스텀 transform으로 **메모리 내 처리** (원본 파일 보존)
- 4채널 정규화: `mean=[2110.71, 2048.15, 1828.06, 3291.53]`, `std=[633.93, 523.48, 533.74, 859.42]`
- Loss: BCE+Sigmoid, Optimizer: AdamW lr=6e-5, PolyLR, 320K iter, batch=8

### 결과

| 지표 | 값 |
|------|-----|
| Best mIoU | **98.57** |

---

## Mission 4 – 멀티모달 융합 분할

### Mission 4-1: LandSat 전이학습 사전학습

**데이터:** LandSat 8/9 4채널 TIFF (256×256)
**목적:** Mission 4-2 backbone 초기화를 위한 MiT-B5 사전학습

- LS30 정규화: `mean=[10707.62, 10662.64, 9663.04, 15352.7]`, `std=[1658.74, 1259.81, 1148.46, 3358.93]`
- 75K iter (early stop), batch=16

| 지표 | 값 |
|------|-----|
| Pretrain mIoU | **92.77** |

---

### Mission 4-2: CMNeXt 기반 멀티모달 융합 분할

**데이터:**
- 주 입력: Sentinel-2 RGBN (4채널, 512×512)
- 보조 입력: 환경 데이터 48채널 (CO × 12개월 + NO2 × 12 + SO2 × 12 + GEMS × 12)

**모델:** CMNeXt + SegFormer MiT-B5

### 모델 구조
```
RGBN 입력 (4ch)
  └─ 1×1 Conv Adapter → MiT-B5 (S1 ~ S4)
                              ├─ S1 (융합 없음)
                              ├─ S2 ─┐
                              ├─ S3 ─┼─ CMNeXt Fusion Modules
                              └─ S4 ─┘
                                   ├─ SelfQueryHub: 위치별 최적 modal 선택
                                   ├─ PPX: 병렬 풀링 (3,7,11) 멀티스케일 강화
                                   ├─ FRM: 채널+공간 교차 모달 보정
                                   └─ FFM: Cross-Attention 기반 최종 융합
                                        └─ SegformerHead → 산업단지 분할 마스크

ENV 입력 (48ch: CO/NO2/SO2/GEMS × 12개월)
  └─ ENV 인코더 → S2/S3/S4 해상도별 feature
```

### 주요 설계 결정
- **융합 위치: S2~S4만** (S1 제외 – ENV 데이터가 S1 해상도와 불일치)
- Loss: BCE + Lovász (1:1 비율)
- Optimizer: AdamW lr=6e-5, PolyLR, 200K iter, batch=16
- 전이학습 backbone(LandSat 사전학습)으로 초기화

### 결과

| 비교 | mIoU |
|------|------|
| Mission 3 (RGBN only) | 98.57 |
| Mission 4 (RGBN + ENV) | **98.85** |
| 향상 | +0.28 pp |

---

## 환경 설정

### Mission 1 & 2
```bash
pip install -r 5G데이터_예선제출물/mission1/requirements.txt
pip install -r 5G데이터_예선제출물/mission2/requirements.txt
```

### Mission 3
```bash
pip install -r 5G데이터_예선제출물/mission3/requirements.txt
# MMSegmentation v0.5 이상 필요
pip install mmsegmentation mmengine mmcv
```

### Mission 4
```bash
pip install -r 5G데이터_본선제출물/requirements.txt
# MMSegmentation + tifffile 필요
pip install mmsegmentation mmengine mmcv tifffile
```

---

## 전체 결과 요약

| Mission | 태스크 | 모델 | 결과 |
|---------|--------|------|------|
| 1 | 굴뚝 탐지 (BBox) | YOLOv8x | mAP50=0.993 |
| 2 | 굴뚝 높이 추정 | ConvNeXt-Tiny + ROIAlign + CrossAttn | RMSE=4.939m, MAPE=2.90% |
| 3 | 산업단지 분할 | SegFormer MiT-B5 | mIoU=98.57 |
| 4 | 멀티모달 산업단지 분할 | CMNeXt + MiT-B5 | mIoU=98.85 |
