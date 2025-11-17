from langchain.schema import Document
from datetime import date

# 분석가
analyst_form = """
{
    "answer":"문서를 바탕으로 한 추론과 결론",
    "link":[
        "근거로 사용한 문서의 링크 1",
        "근거로 사용한 문서의 링크 2",
        ....
    ]
}
"""


def analyst_prompt(keyword: str, form: str):
    return f"""
당신은 최고의 검색어 분석 전문가 입니다.
입력으로 키워드와 여러 문서들의 내용이 주어집니다.
해당 문서들은 모두 {keyword} 라는 검색어로 검색된 내용들 중 가장 최근의 기사들입니다.
오늘은 {date.today()} 이며, 작성된 날짜 또한 최근에 작성된 기사들입니다. 2024년도가 아닙니다.
아래와 같은 규칙과 방법으로 현재 이 키워드가 왜 트랜디해졌는지 분석하십시오.

근거는 반드시 문서만을 참고하여 주십시오.
각 문서마다 핵심 이슈를 요약하여 유사한 문서가 있는지 확인해야합니다.
여러개의 문서 중 유사한 주제로 3개 이상 동시에 발견되어야합니다.
감정적 단어나 추측적 표현("~일 것이다")은 피하고, 실제 문서 근거를 기반으로 결론을 도출합니다.
문서에 존재하지 않는 문구를 출력하지 않습니다.
입력에서 어떤 문서도 제공되지 않았거나 대다수의 문서가 잘못되었을 경우, 다음 문구를 결론으로 출력하십시오:
- 네이버 뉴스가 올바른 문서를 제공하지 않았습니다.

출력 형식은 다음과 같이 JSON으로 주십시오. 코드 블럭을 사용하지 마십시오: {form}
"""


def analyst_input_prompt(searched_documents: list[Document], news: list[str]) -> str:
    prompt = f"""
다음은 해당 검색어로 검색된 뉴스의 내용입니다. 참고하십시오:
{news}

이 내용은 해당 검색어와 관련되어있는 것으로 보이는 다른 검색어의
뉴스 내용입니다. 필요할 경우 추가적인 추론의 근거로 참고하십시오:
"""
    for document in searched_documents:
        prompt += f"""
- {document.page_content}
    - keyword: {document.metadata["keyword"]}
    - link: {document.metadata["link"]}
"""
    return prompt


# 검증자
validator_form = """
{
    "validation": "yes" 또는 "no",
    "reason": "판단의 이유 한 두 줄로 설명"
}
"""


def validator_prompt(keyword: str, form: str):
    return f"""
당신은 엄격한 리뷰어 입니다.
이 내용은 벡터DB에서 {keyword}로 검색되었습니다.
실제로 해당 내용과 관련이 있는지 간단한 문구와 함께 검증하시오.

다음과 같은 Json 양식으로 한 줄로 작성해야 합니다. 추가적인 정보를 덧붙이지 마십시오. 코드 블럭을 사용하지 마십시오: {form}
"""


# 요약자
summarizer_prompt = "다음 뉴스를 300자 내로 정리하십시오:"

# 분류자
# 먼저 정의된 카테고리 목록
ALLOWED_CATEGORIES = [
    "건강",
    "게임",
    "과학",
    "기술",
    "기타",
    "기후",
    "미용 및 패션",
    "법률 및 정부",
    "반려동물 및 동물",
    "비즈니스 및 금융",
    "쇼핑",
    "스포츠",
    "식음료",
    "엔터테이먼트",
    "자동차",
    "정치",
    "취업 및 교육",
]

classifier_form = """
{
    "summarize":"내용 요약",
    "tags":[
        "태그1",
        "태그2",
        ...
    ]
    "category":"카테고리"
}
"""
classifier_prompt = f"""
해당 내용을 정말 간단히 75자 내로 요약하고, 태그와 카테고리를 생성해주십시오.
출력 양식은 다음과 같이 Json 형식으로 작성해야합니다: {classifier_form}
태그는 아무 단어나 상관 없지만, 카테고리는 다음 중 하나를 선택해야 합니다:
"""
for category in ALLOWED_CATEGORIES:
    classifier_prompt += "- " + category + "\n"

print("load prompts")
