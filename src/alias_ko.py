"""Knowledge-based rewriting of hangul vessel names (no learning, no test data).

Port-call records sometimes write an English ship name by *meaning* ("Morning" -> 아침) or use the
Korean name of the operating company ("KMTC" -> 고려해운). A phonetic matcher cannot link these, so
for such tokens we add a rewritten variant that uses the usual hangul transliteration of the English
word. Rows without such tokens are left exactly as they are.
"""
from __future__ import annotations

import unicodedata

# Korean meaning word -> hangul transliterations of the English word(s) it translates
MEANING = {
    "아침": ["모닝"], "새벽": ["던"], "여명": ["던"], "영광": ["글로리"], "여왕": ["퀸"], "왕": ["킹"], "공주": ["프린세스"],
    "왕자": ["프린스"], "왕관": ["크라운"], "별": ["스타"], "샛별": ["모닝 스타"], "태양": ["썬", "선"], "해님": ["썬"], "달": ["문"],
    "하늘": ["스카이"], "바다": ["씨", "오션"], "해양": ["오션", "마린"], "대양": ["오션"], "태평양": ["퍼시픽"], "대서양": ["아틀란틱"],
    "인도양": ["인디언 오션"], "섬": ["아일랜드"], "강": ["리버"], "호수": ["레이크"], "산": ["마운틴"], "바위": ["록"],
    "다리": ["브릿지", "브리지"], "항구": ["하버", "포트"], "만": ["베이"], "해안": ["코스트"], "곶": ["케이프"],
    "개척자": ["파이오니어", "파이오니아"], "탐험가": ["익스플로러"], "항해자": ["보이저"], "항해사": ["네비게이터"],
    "무역상": ["트레이더"], "상인": ["머천트"], "선장": ["캡틴"], "제독": ["애드미럴"], "기사": ["나이트"], "전사": ["워리어"],
    "희망": ["호프"], "꿈": ["드림"], "승리": ["빅토리"], "정신": ["스피릿"], "영혼": ["소울"], "용기": ["커리지"],
    "자유": ["리버티", "프리덤"], "평화": ["피스"], "조화": ["하모니"], "행운": ["포춘", "럭키"], "기쁨": ["조이"], "은혜": ["그레이스"],
    "신뢰": ["트러스트"], "믿음": ["페이스"], "지혜": ["위즈덤"], "단결": ["유니티"], "안전": ["세이프티"], "보물": ["트레저"],
    "미래": ["퓨처"], "세기": ["센츄리"], "세계": ["글로벌", "월드"], "우주": ["코스모스"], "은하": ["갤럭시"], "행성": ["플래닛"],
    "수평선": ["호라이즌"], "지평선": ["호라이즌"], "햇살": ["선샤인"], "햇빛": ["선샤인"], "무지개": ["레인보우"], "구름": ["클라우드"],
    "바람": ["윈드"], "폭풍": ["스톰"], "번개": ["썬더"], "불꽃": ["플레임"], "빛": ["라이트"], "그림자": ["섀도우"],
    "황금": ["골든"], "금": ["골드"], "은": ["실버"], "진주": ["펄"], "수정": ["크리스탈"], "옥": ["제이드"], "보석": ["주얼"],
    "독수리": ["이글"], "매": ["호크"], "사자": ["라이온"], "호랑이": ["타이거"], "용": ["드래곤"], "불사조": ["피닉스"],
    "돌고래": ["돌핀"], "고래": ["웨일"], "백조": ["스완"], "갈매기": ["씨걸"], "꽃": ["플라워"], "장미": ["로즈"], "백합": ["릴리"],
    "난초": ["오키드"], "봄": ["스프링"], "여름": ["썸머"], "가을": ["어텀"], "겨울": ["윈터"],
    "푸른": ["블루"], "파란": ["블루"], "녹색": ["그린"], "초록": ["그린"], "흰": ["화이트"], "하얀": ["화이트"], "검은": ["블랙"],
    "붉은": ["레드"], "빨간": ["레드"], "위대한": ["그레이트"], "웅장한": ["그랜드"], "밝은": ["브라이트"], "행복한": ["해피"],
    "행복": ["해피"], "부유한": ["리치"], "우수한": ["슈피리어"], "고귀한": ["노블"], "충실한": ["로열"], "영원한": ["이터널"],
    "급행": ["익스프레스"], "선구자": ["파이오니어"], "지도자": ["리더"], "챔피언": ["챔피언"], "도전": ["챌린지"], "동방": ["오리엔트"],
    "동양": ["오리엔트"], "아시아": ["아시아"], "한국": ["코리아"], "중국": ["차이나"], "일본": ["재팬"],
}
# Korean company / brand names -> transliteration of the English prefix used in vessel names
COMPANY = {
    "고려해운": ["케이엠티씨"], "고려": ["케이엠티씨"], "장금상선": ["시노코르"], "장금": ["시노코르"], "현대상선": ["에이치엠엠", "현대"],
    "에이치엠엠": ["현대"], "현대글로비스": ["글로비스"], "팬오션": ["팬 오션", "팬오션"], "범양상선": ["팬 오션"],
    "흥아해운": ["흥아"], "흥아라인": ["흥아"], "남성해운": ["남성"], "천경해운": ["씨케이"], "동영해운": ["피씨엘"],
    "범주해운": ["팬콘"], "태영상선": ["태영"], "대한해운": ["대한"], "에스케이해운": ["에스케이"], "에이치라인": ["에이치 라인"],
    "폴라리스쉬핑": ["폴라리스"], "시노트란스": ["시노트랜스"], "코스코": ["코스코"], "머스크": ["머스크"], "에버그린": ["에버"],
}
_SUFFIX = ("의", "호")
MAX_VARIANTS = 6


def _norm(tok: str) -> list[str]:
    out = [tok]
    for s in _SUFFIX:
        if tok.endswith(s) and len(tok) > len(s) + 1:
            out.append(tok[: -len(s)])
    return out


def rewrite(name: str) -> list[str]:
    """Alternative spellings of a hangul name with meaning words / company names rewritten. [] if none apply."""
    if not isinstance(name, str):
        return []
    name = unicodedata.normalize("NFC", name)
    toks = name.split()
    if not toks:
        return []
    options = []
    hit = False
    for t in toks:
        alts = [t]
        for base in _norm(t):
            if base in MEANING:
                alts += MEANING[base]
            if base in COMPANY:
                alts += COMPANY[base]
        # company name glued to the rest of the name (e.g. 고려해운톈진)
        for comp, rep in COMPANY.items():
            if len(comp) >= 3 and t.startswith(comp) and len(t) > len(comp):
                alts += [r + " " + t[len(comp):] for r in rep]
        alts = list(dict.fromkeys(alts))
        hit = hit or len(alts) > 1
        options.append(alts)
    if not hit:
        return []
    out = []
    full = " ".join(a[1] if len(a) > 1 else a[0] for a in options)
    if full != name:
        out.append(full)
    for i, alts in enumerate(options):
        for a in alts[1:]:
            s = " ".join(toks[:i] + [a] + toks[i + 1:])
            if s != name and s not in out:
                out.append(s)
            if len(out) >= MAX_VARIANTS:
                return out
    return out
