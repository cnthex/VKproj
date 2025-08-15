from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, time
import re
import pytz
from dateutil import tz
from dateparser.search import search_dates  # добавить в верх к импортам
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

# -----------------------------
# Конфигурация и инициализация
# -----------------------------

# В проде можно вынести в env
DEFAULT_TZ = "Europe/Moscow"
MODEL_NAME = "dslim/bert-base-NER"  # предобученная NER-модель на OntoNotes

print("\n[VK-AI] Загружаем NER-пайплайн... это займёт ~несколько секунд при первом запуске")
ner = pipeline("token-classification", model=MODEL_NAME, aggregation_strategy="simple")
print("[VK-AI] NER-пайплайн загружен")

# Демо-каталог пользователей (имя -> таймзона)
USER_DIRECTORY = {
    "alice": "Europe/Moscow",      # UTC+3
    "bob": "Europe/Samara",        # UTC+4
    "carol": "Asia/Yekaterinburg", # UTC+5
    "design@team": "Europe/Moscow",
    "managers@team": "Europe/Moscow",
}

# Демо-занятость (busy) — список занятых интервалов в локальном TZ пользователя
# В реальной интеграции заменить на запрос к календарю VK WorkSpace/VK Teams
BUSY_CALENDARS: Dict[str, List[Tuple[datetime, datetime]]] = {}

now_msk = datetime.now(pytz.timezone(DEFAULT_TZ))

def block(user: str, start: datetime, end: datetime):
    BUSY_CALENDARS.setdefault(user, []).append((start, end))

# Сгенерируем пример занятости на ближайшие 7 дней (будни, рабочее время)
for user, tzname in USER_DIRECTORY.items():
    tzinfo = pytz.timezone(tzname)
    base = datetime.now(tzinfo).replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(1, 6):  # 5 рабочих дней
        day = base + timedelta(days=d)
        # Встреча 10:00–10:30
        block(user, day.replace(hour=10, minute=0), day.replace(hour=10, minute=30))
        # Встреча 13:00–13:45
        block(user, day.replace(hour=13, minute=0), day.replace(hour=13, minute=45))
        # Встреча 16:00–17:00
        block(user, day.replace(hour=16, minute=0), day.replace(hour=17, minute=0))

# -----------------------------
# Модели запросов/ответов API
# -----------------------------

class SuggestRequest(BaseModel):
    text: str = Field(..., description="Естественно-языковый запрос")
    participants: Optional[List[str]] = Field(None, description="Список участников (логины/группы)")
    duration_min: int = Field(30, description="Длительность встречи в минутах")
    earliest: Optional[str] = Field(None, description="Минимальная дата/время ISO (опц.)")
    latest: Optional[str] = Field(None, description="Максимальная дата/время ISO (опц.)")

class Slot(BaseModel):
    start_utc: str
    end_utc: str
    start_local: Dict[str, str]
    score: float

class SuggestResponse(BaseModel):
    suggestions: List[Slot]
    parsed: Dict[str, Any]

# -----------------------------
# Утилиты: работа со временем
# -----------------------------

MORNING = (time(9, 0), time(12, 0))
AFTERNOON = (time(13, 0), time(17, 0))
EVENING = (time(18, 0), time(21, 0))

KEYWORDS_TO_WINDOWS = {
    "утром": MORNING,
    "morning": MORNING,
    "после обеда": AFTERNOON,
    "днем": AFTERNOON,
    "afternoon": AFTERNOON,
    "вечером": EVENING,
    "evening": EVENING,
}

WEEKDAYS_RU = {
    "понедельник": 0,
    "вторник": 1,
    "среда": 2,
    "четверг": 3,
    "пятница": 4,
    "суббота": 5,
    "воскресенье": 6,
}


def normalize_date(text: str, base_tz: str = DEFAULT_TZ) -> List[datetime]:
    """Пытаемся превратить временные выражения в конкретные даты.
    Используем dateparser (понимает относительные выражения) и эвристику по будням.
    Возвращаем список кандидатов-дней (без времени).
    """
    candidates = []
    # 1) Явные даты через dateparser
    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now(pytz.timezone(base_tz))}
    parsed = search_dates(text, settings=settings, languages=["ru", "en"]) or []
    for _, dt in parsed:
        candidates.append(dt)

    # 2) Ключевые слова дней недели ("в пятницу")
    for wd_ru, idx in WEEKDAYS_RU.items():
        if re.search(fr"\b{wd_ru}\b", text, re.IGNORECASE):
            today = datetime.now(pytz.timezone(base_tz))
            delta = (idx - today.weekday()) % 7
            if delta == 0:
                delta = 7  # следующее соответствие
            candidates.append(today + timedelta(days=delta))

    # 3) Если ничего не нашли — взять ближайшие 5 рабочих дней
    if not candidates:
        today = datetime.now(pytz.timezone(base_tz))
        for d in range(1, 6):
            candidates.append(today + timedelta(days=d))

    # Нормализуем к полуночи TZ
    normed = []
    for dt in candidates:
        tzinfo = dt.tzinfo or pytz.timezone(base_tz)
        loc = dt.astimezone(tzinfo)
        loc0 = loc.replace(hour=0, minute=0, second=0, microsecond=0)
        normed.append(loc0)
    # Уникализируем
    uniq = []
    seen = set()
    for d in normed:
        k = d.date().isoformat()
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return sorted(uniq)


def preferred_window(text: str) -> Tuple[time, time]:
    for k, (s, e) in KEYWORDS_TO_WINDOWS.items():
        if k in text.lower():
            return s, e
    # дефолт: рабочее окно
    return time(9, 0), time(18, 0)


# -----------------------------
# Извлечение сущностей
# -----------------------------

def extract_entities(text: str) -> Dict[str, Any]:
    """Извлечение дат/организаций/персон из текста с помощью предобученной NER.
    Возвращает словарь с сущностями и нормализованными интервалами времени.
    """
    ents = ner(text)
    dates = [e["word"] for e in ents if e["entity_group"] == "DATE"]
    orgs = [e["word"] for e in ents if e["entity_group"] in ("ORG",)]
    persons = [e["word"] for e in ents if e["entity_group"] in ("PER", "PERSON")]

    # Попытка выделить участников из @упоминаний или явного списка
    mentions = re.findall(r"@([\w\.-]+)", text)
    # Простая эвристика по ключевым словам (дизайнеры/менеджеры) → групповые аккаунты
    groups = []
    if re.search(r"дизайнер", text, re.IGNORECASE):
        groups.append("design@team")
    if re.search(r"менеджер", text, re.IGNORECASE):
        groups.append("managers@team")

    # Окно предпочтений по ключевым словам
    win_start, win_end = preferred_window(text)

    # Кандидатные дни
    days = normalize_date(text, DEFAULT_TZ)

    return {
        "raw_entities": ents,
        "dates_raw": dates,
        "orgs_raw": orgs,
        "persons_raw": persons,
        "mentions": mentions,
        "group_hints": groups,
        "preferred_window": {"start": win_start.isoformat(), "end": win_end.isoformat()},
        "candidate_days": [d.isoformat() for d in days],
    }


# -----------------------------
# Поиск свободных слотов
# -----------------------------

def busy_for_user(user: str) -> List[Tuple[datetime, datetime]]:
    return BUSY_CALENDARS.get(user, [])


def to_utc(dt_local: datetime, tzname: str) -> datetime:
    tzinfo = pytz.timezone(tzname)
    if dt_local.tzinfo is None:
        dt_local = tzinfo.localize(dt_local)
    return dt_local.astimezone(pytz.UTC)


def from_utc(dt_utc: datetime, tzname: str) -> datetime:
    tzinfo = pytz.timezone(tzname)
    return dt_utc.astimezone(tzinfo)


def free_windows_for_day(user: str, day_local: datetime, pref_start: time, pref_end: time) -> List[Tuple[datetime, datetime]]:
    tzname = USER_DIRECTORY.get(user, DEFAULT_TZ)
    tzinfo = pytz.timezone(tzname)
    day0 = day_local.astimezone(tzinfo).replace(hour=0, minute=0, second=0, microsecond=0)
    work_start = datetime.combine(day0.date(), pref_start, tzinfo)
    work_end = datetime.combine(day0.date(), pref_end, tzinfo)

    busy = [(s.astimezone(tzinfo), e.astimezone(tzinfo)) for s, e in busy_for_user(user)
            if s.date() == day0.date()]  # занятости в этот день

    # Начинаем со всего рабочего окна, вычитаем занятость
    free = [(work_start, work_end)]
    for bs, be in sorted(busy):
        new_free = []
        for fs, fe in free:
            if be <= fs or bs >= fe:
                new_free.append((fs, fe))
            else:
                if bs > fs:
                    new_free.append((fs, bs))
                if be < fe:
                    new_free.append((be, fe))
        free = [(s, e) for s, e in new_free if (e - s).total_seconds() >= 5 * 60]
    return free


def intersect_multi(windows_list: List[List[Tuple[datetime, datetime]]]) -> List[Tuple[datetime, datetime]]:
    """Пересечение множественных окон (в UTC)."""
    # Преобразуем все окна к UTC
    windows_list_utc = []
    for user_windows, user in windows_list:
        tzname = USER_DIRECTORY.get(user, DEFAULT_TZ)
        w_utc = [(to_utc(s, tzname), to_utc(e, tzname)) for s, e in user_windows]
        windows_list_utc.append(sorted(w_utc))

    # Алгоритм «скользящих» интервалов
    if not windows_list_utc:
        return []
    current = windows_list_utc[0]
    for other in windows_list_utc[1:]:
        i, j = 0, 0
        merged = []
        while i < len(current) and j < len(other):
            a_start, a_end = current[i]
            b_start, b_end = other[j]
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            if start < end:
                merged.append((start, end))
            if a_end < b_end:
                i += 1
            else:
                j += 1
        current = merged
        if not current:
            break
    return current


def slice_into_slots(intervals: List[Tuple[datetime, datetime]], duration: timedelta) -> List[Tuple[datetime, datetime]]:
    slots = []
    for s, e in intervals:
        t = s
        while t + duration <= e:
            slots.append((t, t + duration))
            t += timedelta(minutes=15)  # шаг скольжения
    return slots


def score_slot(slot: Tuple[datetime, datetime], users: List[str]) -> float:
    """Простая метрика: средняя «удалённость» от края рабочего окна участников (чем ближе к центру дня, тем выше).
    Можно усложнить позже.
    """
    start_utc, end_utc = slot
    scores = []
    for u in users:
        tzname = USER_DIRECTORY.get(u, DEFAULT_TZ)
        local_start = from_utc(start_utc, tzname)
        # Центр рабочего дня 13:30
        center = local_start.replace(hour=13, minute=30)
        dist = abs((local_start - center).total_seconds()) / 3600.0
        scores.append(max(0.0, 5.0 - dist))
    return sum(scores) / len(scores)


def suggest_slots(text: str, participants: List[str], duration_min: int,
                  earliest: Optional[datetime] = None, latest: Optional[datetime] = None) -> Tuple[List[Tuple[datetime, datetime]], Dict[str, Any]]:
    parsed = extract_entities(text)

    # Объединяем участников: из входа + из @упоминаний + групповые подсказки
    detected = parsed["mentions"] + parsed["group_hints"]
    users = list({u for u in (participants or []) + detected if u in USER_DIRECTORY})
    if not users:
        # дефолт — текущий пользователь и один коллега (для демо)
        users = ["alice", "bob"]

    # Формируем диапазон дат
    base_days = [datetime.fromisoformat(d) for d in parsed["candidate_days"]]
    if earliest:
        earliest_dt = dateparser.parse(earliest)
        base_days = [d for d in base_days if d >= earliest_dt]
    if latest:
        latest_dt = dateparser.parse(latest)
        base_days = [d for d in base_days if d <= latest_dt]

    pref_start_s = parsed["preferred_window"]["start"]
    pref_end_s = parsed["preferred_window"]["end"]
    pref_start, pref_end = time.fromisoformat(pref_start_s), time.fromisoformat(pref_end_s)

    # Окна по каждому пользователю > пересечение > слоты нужной длительности
    all_slots_utc: List[Tuple[datetime, datetime]] = []

    for day in base_days[:7]:  # ограничим поиск ближайшей неделей
        per_user_windows = []
        for u in users:
            w = free_windows_for_day(u, day, pref_start, pref_end)
            per_user_windows.append((w, u))
        common = intersect_multi(per_user_windows)
        all_slots_utc.extend(slice_into_slots(common, timedelta(minutes=duration_min)))

    # Сортируем по скору
    scored = [(slot, score_slot(slot, users)) for slot in all_slots_utc]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = [s for s, _ in scored[:5]]
    return top, {"users_considered": users, **parsed}


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(title="VK AI Meeting Helper", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse-intent")
def parse_intent(req: SuggestRequest):
    parsed = extract_entities(req.text)
    return parsed

@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    earliest = dateparser.parse(req.earliest) if req.earliest else None
    latest = dateparser.parse(req.latest) if req.latest else None

    slots, parsed = suggest_slots(
        text=req.text,
        participants=req.participants or [],
        duration_min=req.duration_min,
        earliest=earliest,
        latest=latest,
    )

    # Готовим выдачу: UTC + локальное время для каждого пользователя
    out_slots: List[Slot] = []
    for s_utc, e_utc in slots:
        # строки ISO в UTC
        s_iso = s_utc.astimezone(pytz.UTC).isoformat()
        e_iso = e_utc.astimezone(pytz.UTC).isoformat()
        local_map = {}
        for u in parsed["users_considered"]:
            tzname = USER_DIRECTORY.get(u, DEFAULT_TZ)
            local_map[u] = {
                "tz": tzname,
                "start": from_utc(s_utc, tzname).isoformat(),
                "end": from_utc(e_utc, tzname).isoformat(),
            }
        sc = score_slot((s_utc, e_utc), parsed["users_considered"])
        out_slots.append(Slot(start_utc=s_iso, end_utc=e_iso, start_local=local_map, score=sc))

    return SuggestResponse(suggestions=out_slots, parsed=parsed)


# -----------------------------
# Локальный тест без сервера
# -----------------------------
if __name__ == "__main__":
    demo_text = "Хочу встречу с @alice и менеджерами в пятницу после обеда на 30 минут"
    print("\n[DEMO]", demo_text)
    slots, parsed = suggest_slots(demo_text, participants=["bob"], duration_min=30)
    for i, (s, e) in enumerate(slots, 1):
        print(f"{i}) {s.astimezone(pytz.timezone(DEFAULT_TZ))} — {e.astimezone(pytz.timezone(DEFAULT_TZ))}")