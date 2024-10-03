from datetime import *
import numpy as np

class DateProperties:
    def __init__(self, date: datetime) -> None:
        self._date = date
    
    @property
    def isWeekend(self) -> bool:
        return self._date.weekday() in [5, 6]
    
    @property
    def getTime(self) -> tuple:
        return self._date.hour, self._date.minute, self._date.second
    
    @property
    def getDate(self) -> tuple:
        return self._date.year, self._date.month, self._date.day
    
    def betweenTime(self, start: str, end: str) -> bool:
        start_time = datetime.strptime(start, "%H:%M:%S").time()
        end_time = datetime.strptime(end, "%H:%M:%S").time()
        current_time = self._date.time()
        return start_time <= current_time <= end_time
    
    def betweenDates(self, start: int, end: int) -> bool:
        start_date = datetime.fromtimestamp(start).date()
        end_date = datetime.fromtimestamp(end).date()
        current_date = self._date.date()
        return start_date <= current_date <= end_date
    
    
    @property
    def isFuture(self) -> bool:
        return self._date > datetime.now()
    
    def __call__(self) -> datetime:
        return self._date