# 한국어 리뷰 감성 분석 및 검색 파이프라인
삼성/애플 리뷰 데이터를 활용한 4-class 감성 분류 모델과 비즈니스 인사이트 도출을 위한 검색 파이프라인 구축 | 이화여자대학교 소셜인텔리전스 2026


## 프로젝트 소개
TASK 1
- 한국어 리뷰 4-class 감성 분류 (매우부정/부정/긍정/매우긍정)
- klue/roberta-base 파인튜닝
- 평가 지표: QWK (Quadratic Weighted Kappa)

TASK 2-1
- BM25 + Dense Retrieval + Reranking 기반 검색 시스템 구축
- GPT 기반 리뷰 enrichment (15개 필드)

TASK 2-2
- 삼성/애플 리뷰 데이터 비교 분석
- 소비자 인식 및 구매 패턴 인사이트 도출


## 기술 스택
- **Model**: klue/roberta-base, BM25, Dense Retrieval
- **Language**: Python
- **Tools**: PyTorch, HuggingFace Transformers, OpenAI API


## 주요 기능
